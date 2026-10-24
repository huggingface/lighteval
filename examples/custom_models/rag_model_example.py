# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import re
import string
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lighteval.models.rag.rag_model import (
    GeneratorProtocol,
    RAGAdapterModel,
    RetrievedDocument,
    RetrieverProtocol,
)


logger = logging.getLogger(__name__)


class SimpleVectorRetriever(RetrieverProtocol):
    """Simple in-memory vector retriever using sentence transformers for embeddings."""

    def __init__(self, documents: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(documents, show_progress_bar=False)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0 or not np.any(doc_norms > 0):
            similarities = np.zeros(len(self.documents), dtype=float)
        else:
            safe_doc_norms = doc_norms.copy()
            zero_doc_mask = safe_doc_norms == 0
            safe_doc_norms[zero_doc_mask] = 1.0

            similarities = np.dot(self.embeddings, query_embedding) / (safe_doc_norms * query_norm)
            similarities[zero_doc_mask] = 0.0

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [RetrievedDocument(text=self.documents[idx], score=float(similarities[idx])) for idx in top_indices]


class SimpleGenerator(GeneratorProtocol):
    """Simple generator using a T5 model from transformers."""

    def __init__(self, model_name: str = "google/flan-t5-small", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_answer(self, text: str) -> str:
        """Extract the core answer from generated text by removing common prefixes and phrases.

        Args:
            text: The generated text that may contain extra words.

        Returns:
            The extracted answer, cleaned of common answer prefixes and normalized to match
            TriviaQA gold answer format (lowercase, no punctuation).
        """
        text = text.strip()

        patterns = [
            r"^the answer is\s+(.+)$",
            r"^answer is\s+(.+)$",
            r"^answer:\s*(.+)$",
            r"^the answer:\s*(.+)$",
            r"^it is\s+(.+)$",
            r"^it's\s+(.+)$",
            r"^that is\s+(.+)$",
            r"^that's\s+(.+)$",
            r"^(.+)\s+is the answer$",
            r"^(.+)\s+is correct$",
        ]

        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()

                extracted = re.sub(r"[.,;:!?]+$", "", extracted)
                text = extracted
                break

        normalized = text.lower().translate(str.maketrans("", "", string.punctuation))
        return normalized.strip()

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None, stop_sequences: Optional[list[str]] = None, **kwargs
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        max_new_tokens = max_new_tokens or 50

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]

        extracted_answer = self._extract_answer(generated_text)

        return extracted_answer


class ExampleRAGModel(RAGAdapterModel):
    """Example RAG model implementation using SimpleVectorRetriever and SimpleGenerator."""

    def __init__(self, config):
        documents = self._load_sample_documents()
        retriever = SimpleVectorRetriever(documents)
        generator = SimpleGenerator()
        super().__init__(config, retriever, generator, top_k=3)

    def _load_sample_documents(self) -> list[str]:
        """Load a diverse corpus of trivia documents covering multiple knowledge domains."""
        return [
            "Paris is the capital and most populous city of France.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "France is a country in Western Europe with a population of over 67 million people.",
            "The Louvre Museum is the world's largest art museum and a historic monument in Paris.",
            "The United States of America is a country in North America with 50 states.",
            "Washington D.C. is the capital city of the United States.",
            "The Statue of Liberty is a neoclassical sculpture on Liberty Island in New York Harbor.",
            "Canada is a country in North America with Ottawa as its capital city.",
            "Lester Bowles Pearson became Prime Minister of Canada in April 1963.",
            "In April 1963, Lester Bowles Pearson became Prime Minister of Canada.",
            "Toronto is the largest city in Canada by population.",
            "Vancouver is a major city in British Columbia, Canada.",
            "Montreal is the second-largest city in Canada.",
            "Quebec City is the capital of the province of Quebec in Canada.",
            "London is the capital and largest city of England and the United Kingdom.",
            "Tokyo is the capital and largest city of Japan.",
            "Beijing is the capital city of China.",
            "Moscow is the capital and largest city of Russia.",
            "Berlin is the capital and largest city of Germany.",
            "Rome is the capital city of Italy.",
            "Madrid is the capital and largest city of Spain.",
            "Sydney is the largest city in Australia.",
            "Cairo is the capital and largest city of Egypt.",
            "Brasilia is the capital city of Brazil.",
            "New Delhi is the capital of India.",
            "Seoul is the capital and largest city of South Korea.",
            "Mexico City is the capital and largest city of Mexico.",
            "Buenos Aires is the capital and largest city of Argentina.",
            "The French Revolution was a period of radical political and societal change in France from 1789 to 1799.",
            "The American Civil War was fought between 1861 and 1865.",
            "The Declaration of Independence was adopted on July 4, 1776.",
            "World War I lasted from 1914 to 1918.",
            "World War II lasted from 1939 to 1945.",
            "The Berlin Wall fell in November 1989.",
            "The Renaissance was a period of cultural rebirth in Europe from the 14th to the 17th century.",
            "The Industrial Revolution began in Great Britain in the late 18th century.",
            "The Apollo 11 mission landed the first humans on the Moon in July 1969.",
            "The fall of the Roman Empire occurred in 476 AD.",
            "The Magna Carta was signed in 1215.",
            "The Battle of Hastings took place in 1066.",
            "The Black Death pandemic occurred in Europe during the 14th century.",
            "The American Revolution began in 1775 and ended in 1783.",
            "The Russian Revolution occurred in 1917.",
            "The Cold War lasted from approximately 1947 to 1991.",
            "The Vietnam War lasted from 1955 to 1975.",
            "The Korean War lasted from 1950 to 1953.",
            "The Gulf War took place in 1991.",
            "The September 11 attacks occurred in 2001.",
            "Winston Churchill was Prime Minister of the United Kingdom during World War II.",
            "Franklin D. Roosevelt was President of the United States from 1933 to 1945.",
            "Abraham Lincoln was the 16th President of the United States.",
            "Napoleon Bonaparte was Emperor of France from 1804 to 1814.",
            "Julius Caesar was a Roman general and statesman who was assassinated in 44 BC.",
            "Queen Elizabeth II was the longest-reigning British monarch, from 1952 to 2022.",
            "Nelson Mandela was the first black President of South Africa, serving from 1994 to 1999.",
            "Mahatma Gandhi was a leader of the Indian independence movement.",
            "Martin Luther King Jr. was a civil rights leader in the United States.",
            "John F. Kennedy was the 35th President of the United States, assassinated in 1963.",
            "Ronald Reagan was the 40th President of the United States, serving from 1981 to 1989.",
            "Gerald Ford was the 38th President of the United States, born Lesley Lynch King Jr.",
            "Gerald Ford was born Lesley Lynch King Jr. and later changed his name.",
            "Richard Nixon was the 37th President of the United States, serving from 1969 to 1974.",
            "Jimmy Carter was the 39th President of the United States, serving from 1977 to 1981.",
            "George Washington was the first President of the United States, serving from 1789 to 1797.",
            "Thomas Jefferson was the third President of the United States, serving from 1801 to 1809.",
            "Theodore Roosevelt was the 26th President of the United States, serving from 1901 to 1909.",
            "Woodrow Wilson was the 28th President of the United States, serving from 1913 to 1921.",
            "Harry S. Truman was the 33rd President of the United States, serving from 1945 to 1953.",
            "Dwight D. Eisenhower was the 34th President of the United States, serving from 1953 to 1961.",
            "Lyndon B. Johnson was the 36th President of the United States, serving from 1963 to 1969.",
            "Bill Clinton was the 42nd President of the United States, serving from 1993 to 2001.",
            "Barack Obama was the 44th President of the United States, serving from 2009 to 2017.",
            "Donald Trump was the 45th President of the United States, serving from 2017 to 2021.",
            "Margaret Thatcher was the first female Prime Minister of the United Kingdom, serving from 1979 to 1990.",
            # Science - Important Facts Only
            "DNA stands for deoxyribonucleic acid and contains genetic information.",
            "Evolution is the process by which species change over time through natural selection.",
            "Light travels at approximately 299,792,458 meters per second in a vacuum.",
            "Einstein's theory of relativity includes E equals mc squared.",
            "Gravity is the force that attracts objects with mass toward each other.",
            "The speed of sound in air is approximately 343 meters per second.",
            "Atoms are composed of protons, neutrons, and electrons.",
            "The periodic table organizes chemical elements by atomic number.",
            "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
            "The Earth orbits the Sun approximately every 365.25 days.",
            "The Moon orbits the Earth approximately every 27.3 days.",
            "William Shakespeare wrote plays including Hamlet, Romeo and Juliet, and Macbeth.",
            "Charles Dickens wrote novels including A Tale of Two Cities and Great Expectations.",
            "Jane Austen wrote Pride and Prejudice and Sense and Sensibility.",
            "Mark Twain wrote The Adventures of Tom Sawyer and Adventures of Huckleberry Finn.",
            "George Orwell wrote 1984 and Animal Farm.",
            "J.K. Rowling wrote the Harry Potter series of books.",
            "J.R.R. Tolkien wrote The Lord of the Rings and The Hobbit.",
            "Ernest Hemingway wrote The Old Man and the Sea and For Whom the Bell Tolls.",
            "F. Scott Fitzgerald wrote The Great Gatsby.",
            "Harper Lee wrote To Kill a Mockingbird.",
            "Herman Melville wrote Moby-Dick.",
            "Leo Tolstoy wrote War and Peace and Anna Karenina.",
            "Victor Hugo wrote Les Misérables and The Hunchback of Notre-Dame.",
            "The FIFA World Cup is held every four years.",
            "The Olympic Games are held every four years, alternating between Summer and Winter Olympics.",
            "Soccer, also known as football, is the world's most popular sport.",
            "Basketball was invented by James Naismith in 1891.",
            "The Super Bowl is the championship game of the National Football League.",
            "Michael Jordan is widely considered the greatest basketball player of all time.",
            "Pelé is considered one of the greatest soccer players of all time.",
            "The Tour de France is an annual bicycle race held in France.",
            "Swimming includes four main strokes: freestyle, backstroke, breaststroke, and butterfly.",
            # Entertainment - General
            "The Academy Awards, also known as the Oscars, honor achievements in film.",
            "The Beatles were a British rock band formed in Liverpool in 1960.",
            "Elvis Presley was known as the King of Rock and Roll.",
            "Michael Jackson was known as the King of Pop.",
            "Mozart was a prolific composer of the Classical period.",
            "Beethoven composed nine symphonies, including his famous Ninth Symphony.",
            "The Great Wall of China is one of the Seven Wonders of the World.",
            "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
            "The Amazon River is the largest river by discharge volume in the world.",
            "The Sahara Desert is the largest hot desert in the world.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "The Eiffel Tower was completed in 1889 for the Exposition Universelle.",
            "The Statue of Liberty was a gift from France to the United States.",
            "The Great Pyramid of Giza is the oldest of the Seven Wonders of the Ancient World.",
            "The Colosseum in Rome is an ancient amphitheater built in the first century AD.",
            "The Taj Mahal is a mausoleum in India built by Shah Jahan.",
        ]

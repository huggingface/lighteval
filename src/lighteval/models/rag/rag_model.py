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

from dataclasses import dataclass
from typing import Optional, Protocol

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached


@dataclass
class RetrievedDocument:
    """Represents a document retrieved by a retriever.

    Attributes:
        text (str): The text content of the retrieved document.
        score (float, optional): Retrieval score/relevance score. Higher is better.
        metadata (dict, optional): Additional metadata about the document (e.g., doc_id, source).
    """

    text: str
    score: Optional[float] = None
    metadata: Optional[dict] = None


class RetrieverProtocol(Protocol):
    """Protocol defining the interface for retrieval components in RAG systems.

    Any class implementing this protocol can be used as a retriever in RAGAdapterModel.
    This allows maximum flexibility - retrievers can use FAISS, ColBERT, dense embeddings,
    sparse retrieval, or any other retrieval method.

    Example implementations:
        - FAISS-based vector search
        - BM25 sparse retrieval
        - Hybrid dense+sparse retrieval
        - Latent space retrieval (for latent RAG systems)
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve relevant documents for a given query.

        Args:
            query (str): The query string to retrieve documents for.
            top_k (int): Number of documents to retrieve. Defaults to 5.

        Returns:
            list[RetrievedDocument]: List of retrieved documents, ordered by relevance (highest first).
        """
        ...


class GeneratorProtocol(Protocol):
    """Protocol defining the interface for generation components in RAG systems.

    Any class implementing this protocol can be used as a generator in RAGAdapterModel.
    This allows maximum flexibility - generators can be Transformers models, vLLM, TGI endpoints,
    or any other text generation system.

    Example implementations:
        - TransformersModel wrapped for generation
        - vLLM client
        - TGI endpoint client
        - Custom generation logic
    """

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None, stop_sequences: Optional[list[str]] = None, **kwargs
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt (str): The input prompt (may include retrieved context).
            max_new_tokens (int, optional): Maximum number of tokens to generate.
            stop_sequences (list[str], optional): Sequences that should stop generation.
            **kwargs: Additional generation parameters (temperature, top_p, etc.).

        Returns:
            str: The generated text.
        """
        ...


class ContextFormatter:
    """Utility class for formatting retrieved context with queries.

    Provides common patterns for combining retrieved documents with the original query.
    """

    @staticmethod
    def format_context_separator(query: str, retrieved_docs: list[RetrievedDocument], separator: str = "\n\n") -> str:
        """Format context by joining retrieved documents with a separator.

        Args:
            query (str): The original query.
            retrieved_docs (list[RetrievedDocument]): Retrieved documents to include.
            separator (str): Separator between documents. Defaults to "\n\n".

        Returns:
            str: Formatted prompt with context prepended to query.
        """
        context_parts = [doc.text for doc in retrieved_docs]
        context = separator.join(context_parts)
        return f"{context}{separator}{query}"

    @staticmethod
    def format_context_instruction(
        query: str,
        retrieved_docs: list[RetrievedDocument],
        instruction_template: str = "Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}",
    ) -> str:
        """Format context using a custom instruction template.

        Args:
            query (str): The original query.
            retrieved_docs (list[RetrievedDocument]): Retrieved documents to include.
            instruction_template (str): Template string with {context} and {query} placeholders.

        Returns:
            str: Formatted prompt using the template.
        """
        context = "\n\n".join([doc.text for doc in retrieved_docs])
        return instruction_template.format(context=context, query=query)

    @staticmethod
    def format_context_numbered(query: str, retrieved_docs: list[RetrievedDocument], prefix: str = "Context") -> str:
        """Format context with numbered documents.

        Args:
            query (str): The original query.
            retrieved_docs (list[RetrievedDocument]): Retrieved documents to include.
            prefix (str): Prefix for numbered items. Defaults to "Context".

        Returns:
            str: Formatted prompt with numbered context.
        """
        context_parts = [f"{prefix} {i + 1}: {doc.text}" for i, doc in enumerate(retrieved_docs)]
        context = "\n\n".join(context_parts)
        return f"{context}\n\n{query}"


class RAGAdapterModel(LightevalModel):
    """Base class for RAG (Retrieval-Augmented Generation) model adapters.

    This class provides a standardized way to evaluate RAG systems on LightEval benchmarks.
    It implements the LightevalModel interface by composing a retriever and generator.

    The adapter works with any existing LightEval benchmark because it:
    1. Receives standard Doc objects (same as any other model)
    2. Performs retrieval internally using the query
    3. Augments the prompt with retrieved context
    4. Generates a response using the generator
    5. Returns standard ModelResponse objects

    This allows RAG systems to be evaluated on benchmarks like MMLU, GSM8K, TriviaQA, etc.
    using the same metrics (exact_match, F1, ROUGE, etc.) as standard language models.

    Attributes:
        retriever (RetrieverProtocol): The retrieval component.
        generator (GeneratorProtocol): The text generation component.
        context_formatter (ContextFormatter): Utility for formatting context with queries.
        top_k (int): Number of documents to retrieve per query. Defaults to 5.
        include_retrieval_metadata (bool): Whether to include retrieval info in ModelResponse.metadata.

    Example:
        ```python
        from lighteval.models.rag.rag_model import RAGAdapterModel, RetrievedDocument

        class MyRetriever:
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
                # Your retrieval logic here
                return [RetrievedDocument(text="...", score=0.95)]

        class MyGenerator:
            def generate(self, prompt: str, **kwargs) -> str:
                # Your generation logic here
                return "Generated text"

        class MyRAGModel(RAGAdapterModel):
            def __init__(self, config):
                retriever = MyRetriever()
                generator = MyGenerator()
                super().__init__(config, retriever, generator)
        ```
    """

    def __init__(
        self,
        config: ModelConfig,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
        top_k: int = 5,
        include_retrieval_metadata: bool = True,
        context_formatter: Optional[ContextFormatter] = None,
    ):
        """Initialize the RAG adapter model.

        Args:
            config (ModelConfig): Model configuration (from CustomModelConfig or similar).
            retriever (RetrieverProtocol): The retrieval component.
            generator (GeneratorProtocol): The text generation component.
            top_k (int): Number of documents to retrieve per query. Defaults to 5.
            include_retrieval_metadata (bool): Whether to include retrieval info in responses.
                Defaults to True.
            context_formatter (ContextFormatter, optional): Custom formatter. If None, uses
                default separator-based formatting.
        """

        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.include_retrieval_metadata = include_retrieval_metadata
        self.context_formatter = context_formatter or ContextFormatter()

        try:
            self._cache = SampleCache(config)
        except Exception:
            self._cache = None

    @property
    def tokenizer(self):
        """Return a tokenizer from the generator if available.

        Raises:
            AttributeError: If the generator does not provide a tokenizer attribute.
        """
        if hasattr(self.generator, "tokenizer"):
            return self.generator.tokenizer
        raise AttributeError(
            "Generator must provide a tokenizer attribute. "
            "Either implement tokenizer property in your generator or pass a tokenizer explicitly."
        )

    @property
    def add_special_tokens(self) -> bool:
        """Whether to add special tokens during tokenization."""
        if hasattr(self.generator, "add_special_tokens"):
            return self.generator.add_special_tokens
        return True

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length."""
        if hasattr(self.generator, "max_length"):
            return self.generator.max_length
        return 4096

    def _format_prompt_with_context(self, doc: Doc, retrieved_docs: list[RetrievedDocument]) -> str:
        """Format the prompt by combining retrieved context with the query.

        Args:
            doc (Doc): The evaluation document containing the query.
            retrieved_docs (list[RetrievedDocument]): Retrieved documents.

        Returns:
            str: Formatted prompt ready for generation.
        """
        query = doc.query.strip()

        if doc.instruction:
            query = f"{doc.instruction.strip()}\n\n{query}"

        context_text = "\n\n".join([retrieved_doc.text.strip() for retrieved_doc in retrieved_docs])

        if context_text:
            formatted_prompt = f"question: {query} context: {context_text}"
        else:
            formatted_prompt = f"question: {query}"

        return formatted_prompt

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in dataset.splits_iterator():
            for doc in split:
                retrieved_docs = self.retriever.retrieve(doc.query, top_k=self.top_k)
                augmented_prompt = self._format_prompt_with_context(doc, retrieved_docs)

                max_new_tokens = doc.generation_size or self.config.generation_parameters.max_new_tokens
                stop_sequences = doc.stop_sequences or []

                generated_text = self.generator.generate(
                    prompt=augmented_prompt,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=stop_sequences,
                )

                metadata = None
                if self.include_retrieval_metadata:
                    metadata = {
                        "retrieved_docs": [
                            {
                                "text": retrieved_doc.text,
                                "score": retrieved_doc.score,
                                "metadata": retrieved_doc.metadata,
                            }
                            for retrieved_doc in retrieved_docs
                        ],
                        "num_retrieved": len(retrieved_docs),
                    }

                encoded = self.tok_encode(augmented_prompt)
                if isinstance(encoded, list):
                    input_tokens = encoded
                elif hasattr(encoded, "tolist"):
                    input_tokens = encoded.tolist()
                else:
                    input_tokens = list(encoded) if encoded else []

                response = ModelResponse(
                    text=[generated_text],
                    input_tokens=input_tokens,
                    metadata=metadata,
                )
                results.append(response)

        return dataset.get_original_order(results)

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Compute log likelihoods for continuations.

        Note: This requires the generator to support loglikelihood computation.
        Most RAG systems focus on generative evaluation, so this may raise NotImplementedError.

        Args:
            docs (list[Doc]): List of documents with context and continuation pairs.

        Returns:
            list[ModelResponse]: List of responses with log probabilities.

        Raises:
            NotImplementedError: If the generator doesn't support loglikelihood computation.
        """
        if hasattr(self.generator, "loglikelihood"):
            augmented_docs = []
            for doc in docs:
                retrieved_docs = self.retriever.retrieve(doc.query, top_k=self.top_k)
                augmented_prompt = self._format_prompt_with_context(doc, retrieved_docs)

                augmented_doc = Doc(
                    query=augmented_prompt,
                    choices=doc.choices,
                    gold_index=doc.gold_index,
                    instruction=None,
                )
                augmented_docs.append(augmented_doc)

            return self.generator.loglikelihood(augmented_docs)

        raise NotImplementedError(
            "loglikelihood() not supported. The generator must implement loglikelihood() "
            "or use generative evaluation (greedy_until) instead."
        )

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError(
            "loglikelihood_rolling() not supported for RAG systems. "
            "RAG systems are evaluated using generative metrics, not perplexity."
        )

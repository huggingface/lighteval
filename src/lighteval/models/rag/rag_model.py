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
from dataclasses import dataclass, replace
from typing import Optional, Protocol

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached


logger = logging.getLogger(__name__)


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

    **Required Method:**
        - `generate()`: Must be implemented by all generators.

    **Required Attribute or Property:**
        - `tokenizer`: Must be exposed as either an attribute or a read-only property that
          returns a tokenizer object. This is required by RAGAdapterModel for tokenization
          during evaluation. The tokenizer should support the standard tokenizer interface
          (e.g., `encode()`, `decode()` methods). An AttributeError will be raised if the tokenizer is
          not found during runtime.

    **Optional Properties:**
        - `add_special_tokens`: If present, should return a bool indicating whether to add
          special tokens during tokenization. Defaults to True if not present.
        - `max_length`: If present, should return the maximum sequence length supported
          by the generator. Defaults to 4096 if not present.

    **Optional Method:**
        - `loglikelihood()`: If implemented, must match the `LightevalModel.loglikelihood` signature:
          `loglikelihood(docs: list[Doc]) -> list[ModelResponse]`. This is used for logprob-based
          evaluations (e.g., multiple choice tasks). If not implemented, RAGAdapterModel will
          raise NotImplementedError when loglikelihood is called.

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

    @staticmethod
    def format_context_question_t5(query: str, retrieved_docs: list[RetrievedDocument]) -> str:
        """Format context with question first, then context (T5-friendly format).

        This matches the current default behavior for backward compatibility.

        Args:
            query (str): The original query (may already include instruction).
            retrieved_docs (list[RetrievedDocument]): Retrieved documents to include.

        Returns:
            str: Formatted prompt with question first, then context.
        """
        context_text = "\n\n".join([retrieved_doc.text.strip() for retrieved_doc in retrieved_docs])
        if context_text:
            return f"question: {query} context: {context_text}"
        return f"question: {query}"


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
        retrieval_batch_size (int): Batch size for retrieval within splits to avoid memory issues.
            Defaults to RETRIEVAL_BATCH_SIZE (128).

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

    RETRIEVAL_BATCH_SIZE = 128

    def __init__(
        self,
        config: ModelConfig,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
        top_k: int = 5,
        include_retrieval_metadata: bool = True,
        context_formatter: Optional[ContextFormatter] = None,
        retrieval_batch_size: Optional[int] = None,
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
            retrieval_batch_size (int, optional): Batch size for retrieval within splits to avoid
                memory issues. Defaults to RETRIEVAL_BATCH_SIZE (128). Set to None to use default.
        """
        super().__init__()
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.include_retrieval_metadata = include_retrieval_metadata
        self.context_formatter = context_formatter or ContextFormatter()
        self.retrieval_batch_size = retrieval_batch_size or self.RETRIEVAL_BATCH_SIZE

        if not hasattr(self.generator, "tokenizer"):
            raise AttributeError(
                "Generator must provide a tokenizer attribute. "
                "The generator passed to RAGAdapterModel must have a 'tokenizer' attribute "
                "for tokenization during evaluation. Either implement a tokenizer property "
                "in your generator class or ensure your generator exposes a tokenizer attribute."
            )

        try:
            self._cache = SampleCache(config)
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}. Continuing without cache.")
            self._cache = None

    @property
    def tokenizer(self):
        """Return a tokenizer from the generator.

        The tokenizer is validated during initialization to ensure the generator
        provides this attribute. This property assumes the tokenizer exists.

        Returns:
            The tokenizer from the generator.
        """
        return self.generator.tokenizer

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

    def _build_retrieval_query(self, doc: Doc) -> str:
        """Build the query string to use for retrieval, including instruction if present.

        This method constructs the query that will be used to retrieve relevant documents.
        It combines the instruction (if present) with the query text.

        Args:
            doc (Doc): The evaluation document containing the query and optional instruction.

        Returns:
            str: Query string for retrieval (instruction + query if instruction exists, otherwise just query).
        """
        retrieval_query = doc.query.strip()
        if doc.instruction:
            retrieval_query = f"{doc.instruction.strip()}\n\n{retrieval_query}"
        return retrieval_query

    def _retrieve_batch(self, docs: list[Doc]) -> list[list[RetrievedDocument]]:
        """Retrieve documents for a batch of queries.

        This helper method handles batch retrieval, falling back to individual retrieval
        if the retriever doesn't support batch_retrieve. This reduces code duplication
        between greedy_until and loglikelihood methods.

        Args:
            docs (list[Doc]): List of documents to retrieve context for.

        Returns:
            list[list[RetrievedDocument]]: List of retrieved document lists, one per input doc.
        """
        retrieval_queries = [self._build_retrieval_query(doc) for doc in docs]

        if hasattr(self.retriever, "batch_retrieve"):
            batch_retrieved_docs = self.retriever.batch_retrieve(retrieval_queries, top_k=self.top_k)
        else:
            batch_retrieved_docs = [
                self.retriever.retrieve(retrieval_query, top_k=self.top_k) for retrieval_query in retrieval_queries
            ]

        return batch_retrieved_docs

    def _format_prompt_with_context(self, doc: Doc, retrieved_docs: list[RetrievedDocument]) -> str:
        """Format the prompt by combining retrieved context with the query.

        Uses the configured context_formatter to format the prompt. The default
        formatter uses a T5-friendly format, but can be customized for other generators.

        This method first checks for a generic `format_context_question` method on the
        formatter, and falls back to `format_context_question_t5` for backward compatibility.

        Args:
            doc (Doc): The evaluation document containing the query.
            retrieved_docs (list[RetrievedDocument]): Retrieved documents.

        Returns:
            str: Formatted prompt ready for generation.
        """
        query = doc.query.strip()
        if doc.instruction:
            query = f"{doc.instruction.strip()}\n\n{query}"

        formatter = self.context_formatter
        if hasattr(formatter, "format_context_question"):
            return formatter.format_context_question(query, retrieved_docs)

        return formatter.format_context_question_t5(query, retrieved_docs)

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in dataset.splits_iterator():
            split_docs = list(split)

            for start_idx in range(0, len(split_docs), self.retrieval_batch_size):
                batch_docs = split_docs[start_idx : start_idx + self.retrieval_batch_size]
                batch_retrieved_docs = self._retrieve_batch(batch_docs)

                for doc, retrieved_docs in zip(batch_docs, batch_retrieved_docs):
                    augmented_prompt = self._format_prompt_with_context(doc, retrieved_docs)

                    generation_params = getattr(self.config, "generation_parameters", None)
                    config_max_new_tokens = (
                        getattr(generation_params, "max_new_tokens", None) if generation_params is not None else None
                    )
                    max_new_tokens = doc.generation_size or config_max_new_tokens
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
                    if encoded is None:
                        input_tokens = []
                    elif isinstance(encoded, list):
                        input_tokens = encoded
                    elif hasattr(encoded, "tolist"):
                        input_tokens = encoded.tolist()
                    else:
                        try:
                            input_tokens = list(encoded)
                        except TypeError:
                            input_tokens = []

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

        The generator's `loglikelihood` method (if present) must accept `list[Doc]` and return
        `list[ModelResponse]`, matching the `LightevalModel.loglikelihood` signature.

        Args:
            docs (list[Doc]): List of documents with context and continuation pairs.

        Returns:
            list[ModelResponse]: List of responses with log probabilities.

        Raises:
            NotImplementedError: If the generator doesn't support loglikelihood computation.
            TypeError: If the generator's loglikelihood method has an incompatible signature.
        """
        if hasattr(self.generator, "loglikelihood"):
            augmented_docs = []

            for start_idx in range(0, len(docs), self.retrieval_batch_size):
                batch_docs = docs[start_idx : start_idx + self.retrieval_batch_size]
                batch_retrieved_docs = self._retrieve_batch(batch_docs)

                for doc, retrieved_docs in zip(batch_docs, batch_retrieved_docs):
                    augmented_prompt = self._format_prompt_with_context(doc, retrieved_docs)
                    augmented_doc = replace(doc, query=augmented_prompt)
                    augmented_docs.append(augmented_doc)

            try:
                return self.generator.loglikelihood(augmented_docs)
            except TypeError as e:
                raise TypeError(
                    f"Generator's loglikelihood method has an incompatible signature. "
                    f"Expected: loglikelihood(docs: list[Doc]) -> list[ModelResponse]. "
                    f"Error: {e}"
                ) from e

        raise NotImplementedError(
            "loglikelihood() not supported. The generator must implement loglikelihood() "
            "or use generative evaluation (greedy_until) instead."
        )

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError(
            "loglikelihood_rolling() not supported for RAG systems. "
            "RAG systems are evaluated using generative metrics, not perplexity."
        )

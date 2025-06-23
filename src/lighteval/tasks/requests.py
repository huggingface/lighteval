# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

from lighteval.utils.utils import as_list


if TYPE_CHECKING:
    from PIL.Image import Image


class SamplingMethod(str, Enum):
    """
    Enum representing different sampling methods for text generation.
    """

    GENERATIVE = "GENERATIVE"
    LOGPROBS = "LOGPROBS"  # computes logprobs of choices
    PERPLEXITY = "PERPLEXITY"  # computes logprobs of the whole prompt


@dataclass(slots=True)
class Doc:
    """
    Dataclass representing a single evaluation sample or benchmark instance.

    This class encapsulates all the information needed to evaluate a model on a single
    task instance. It contains the input query, expected outputs, metadata, and
    configuration parameters for different types of evaluation tasks.

    The Doc is created from benchmark datasets, where each line/instance becomes one Doc.
    A list of Doc objects is then fed to the model for evaluation.

    **Required Fields (for most tasks):**
        - `query`: The input prompt or question
        - `choices`: Available answer choices (for multiple choice tasks)
        - `gold_index`: Index(es) of the correct answer(s)

    **Optional Fields (task-dependent):**
        - `instruction`: System prompt or task instructions
        - `images`: Visual inputs for multimodal tasks

    Attributes:
        query (str):
            The main query, prompt, or question to be sent to the model.
            This is the primary input that the model will process.
            **Required for**: All evaluation tasks.

        choices (list[str]):
            List of possible answer choices for the query.
            For multiple choice tasks, this contains all options (A, B, C, D, etc.).
            For generative tasks, this may be empty or contain reference answers.
            **Required for**: Multiple choice tasks, classification tasks.

        gold_index (Union[int, list[int]]):
            Index or indices of the correct answer(s) in the choices list.
            For single correct answers, use an integer (e.g., 0 for first choice).
            For multiple correct answers, use a list (e.g., [0, 2] for first and third).
            **Required for**: All tasks that have defined correct answers.

        instruction (str | None):
            System prompt or task-specific instructions to guide the model.
            This is typically prepended to the query to set context or behavior.
            **Used for**: Chat models, instruction-following tasks, system prompts.

        id (str):
            Unique identifier for this evaluation instance.
            Useful for tracking results and debugging.
            **Used for**: Result tracking, error analysis, dataset management.

        images (list["Image"] | None):
            List of PIL Image objects for multimodal tasks.
            **Required for**: Vision-language tasks, image captioning, visual question answering.

        specific (dict | None):
            Task-specific information or metadata.
            Can contain any additional data needed for evaluation.
            **Used for**: Custom metrics, task-specific processing, metadata storage.

        task_name (str):
            Name of the task or benchmark this Doc belongs to.
            **Used for**: Task identification, result organization, metric calculation.

        # Few-shot Learning Parameters
        num_asked_few_shots (int):
            Number of few-shot examples requested for this instance.
            **Used for**: Few-shot learning configuration, example counting.

        num_effective_few_shots (int):
            Actual number of few-shot examples used (may differ from requested).
            **Used for**: Few-shot analysis, performance correlation studies.

        fewshot_samples (list):
            List of Doc objects representing few-shot examples.
            These examples are prepended to the main query to provide context.
            **Required for**: Few-shot learning tasks, in-context learning evaluation.

        sampling_methods (list[SamplingMethod]):
            List of sampling methods to use for this instance.
            Options: GENERATIVE, LOGPROBS, PERPLEXITY.
            **Used for**: Multi-method evaluation, comprehensive model assessment.

        fewshot_sorting_class (Optional[str]):
            Class label for balanced few-shot example selection.
            Used to ensure diverse representation in few-shot examples.
            **Used for**: Balanced few-shot sampling, fair evaluation.

        # PMI and Context Parameters
        unconditioned_query (Optional[str]):
            Query without task-specific context for PMI normalization.
            Used to calculate: log P(choice | Query) - log P(choice | Unconditioned Query).
            **Required for**: PMI-based metrics, bias analysis, context-aware evaluation.

        original_query (str | None):
            The query before any preprocessing or modification.
            **Used for**: Debugging, preprocessing analysis, result interpretation.

        # Generation Control Parameters
        generation_size (int | None):
            Maximum number of tokens to generate for this instance.
            **Used for**: Controlling generation length, preventing infinite loops.

        stop_sequences (list[str] | None):
            List of strings that should stop generation when encountered.
            **Used for**: Controlled generation, preventing unwanted continuations.

        use_logits (bool):
            Whether to return logits (raw model outputs) in addition to text.
            **Used for**: Probability analysis, confidence scoring, detailed evaluation.

        num_samples (int):
            Number of different samples to generate for this instance.
            **Used for**: Diversity analysis, uncertainty estimation, ensemble methods.

        generation_grammar (None):
            Grammar constraints for generation (currently not implemented).
            **Reserved for**: Future structured generation features.

    Methods:
        get_golds():
            Returns the correct answer(s) as strings based on gold_index.
            Handles both single and multiple correct answers.

    Usage Examples:

        **Multiple Choice Question:**
        ```python
        doc = Doc(
            query="What is the capital of France?",
            choices=["London", "Paris", "Berlin", "Madrid"],
            gold_index=1,  # Paris is the correct answer
            instruction="Answer the following geography question:",
        )
        ```

        **Generative Task:**
        ```python
        doc = Doc(
            query="Write a short story about a robot.",
            choices=[],  # No predefined choices for generative tasks
            gold_index=0,  # Not used for generative tasks
            generation_size=100,
            stop_sequences=["\n\n", "The End"],
        )
        ```

        **Few-shot Learning:**
        ```python
        doc = Doc(
            query="Translate 'Hello world' to Spanish.",
            choices=["Hola mundo", "Bonjour monde", "Ciao mondo"],
            gold_index=0,
            fewshot_samples=[
                Doc(query="Translate 'Good morning' to Spanish.",
                    choices=["Buenos días", "Bonjour", "Buongiorno"],
                    gold_index=0),
                Doc(query="Translate 'Thank you' to Spanish.",
                    choices=["Gracias", "Merci", "Grazie"],
                    gold_index=0)
            ],
        )
        ```

        **Multimodal Task:**
        ```python
        doc = Doc(
            query="What is shown in this image?",
            choices=["A cat", "A dog", "A bird"],
            gold_index=0,
            images=[pil_image],  # PIL Image object
        )
        ```

        **PMI Analysis:**
        ```python
        doc = Doc(
            query="The answer to the math problem is: 42",
            choices=["42", "43", "41"],
            gold_index=0,
            unconditioned_query="The answer is:",  # Context-free version
        )
        ```

    Notes:
        - Most fields are optional, but `query`, `choices`, and `gold_index` are required for most tasks
        - The `choices` field can be empty for purely generative tasks
        - `gold_index` can be a single integer or list of integers for multiple correct answers
        - Few-shot examples are themselves Doc objects, creating a nested structure
        - Multimodal tasks require the `images` field to be populated
        - PMI analysis requires both `query` and `unconditioned_query` to be set
        - Task-specific metadata can be stored in the `specific` field
    """

    query: str
    choices: list[str]
    gold_index: Union[int, list[int]]
    instruction: str | None = None  # system prompt to use for the model, if any
    id: str = ""
    images: list["Image"] | None = None  # for multimodal benchmarks
    specific: dict | None = None  # Information which is specific to the current eval

    task_name: str = ""

    # Fewshots parameters
    num_asked_few_shots: int = -1
    num_effective_few_shots: int = -1
    fewshot_samples: list = field(default_factory=list)
    sampling_methods: list[SamplingMethod] = field(default_factory=list)
    fewshot_sorting_class: Optional[str] = None  # class to use to select balanced few-shot samples

    # Uncoditioned query is used for PMI normalization, that's
    # log P(choice | Query) - log P(choice | Unconditioned Query)
    # The uncoditioned query shouldn't contain any information about the task, thus usually it's empty string or 'Answer:'.
    unconditioned_query: Optional[str] = None
    original_query: str | None = None  # the query before preprocessing, if stored

    # Generation parameters
    generation_size: int | None = None  # number of tokens to generate for each sample
    stop_sequences: list[str] | None = None
    use_logits: bool = False  # whether to use logits for the generation or not
    num_samples: int = 1  # number of samples to generate for each sample
    generation_grammar: None = None

    def get_golds(self):
        """Return gold targets extracted from the target dict"""
        gold_indices = as_list(self.gold_index)
        golds = []
        for gold_ix in gold_indices:
            golds.extend(as_list(self.choices[gold_ix]))
        return golds

    def __repr__(self):
        doc_dict = asdict(self)
        return json.dumps(doc_dict)

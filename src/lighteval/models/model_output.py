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

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelResponse:
    """
    A class to represent the response from a model during evaluation.

    This dataclass contains all the information returned by a model during inference,
    including generated text, log probabilities, token information, and metadata.
    Different attributes are required for different types of evaluation metrics.

    Attributes:
        input (str | list | None):
            The original input prompt or context that was fed to the model.
            Used for debugging and analysis purposes.

        text (list[str]):
            The generated text responses from the model. Each element represents
            one generation (useful when num_samples > 1).
            **Required for**: Generative metrics, exact match, llm as a judge, etc.

        logprobs (list[float]):
            Log probabilities of the generated tokens or sequences.
            **Required for**: loglikelihood and perplexity metrics.

        argmax_logits_eq_gold (list[bool]):
            Whether the argmax logits match the gold/expected text.
            Used for accuracy calculations in multiple choice and classification tasks.
            **Required for**: certain loglikelihood metrics.

        unconditioned_logprobs (Optional[list[float]]):
            Log probabilities from an unconditioned model (e.g., without context).
            Used for PMI (Pointwise Mutual Information) normalization.
            **Required for**: PMI metrics.

    Usage Examples:

        **For generative tasks (text completion, summarization):**
        ```python
        response = ModelResponse(
            text=["The capital of France is Paris."],
            input_tokens=[1, 2, 3, 4],
            output_tokens=[[5, 6, 7, 8]]
        )
        ```

        **For multiple choice tasks:**
        ```python
        response = ModelResponse(
            logprobs=[-0.5, -1.2, -2.1, -1.8],  # Logprobs for each choice
            argmax_logits_eq_gold=[False, False, False, False],  # Whether correct choice was selected
            input_tokens=[1, 2, 3, 4],
            output_tokens=[[5], [6], [7], [8]]
        )
        ```

        **For perplexity calculation:**
        ```python
        response = ModelResponse(
            text=["The model generated this text."],
            logprobs=[-1.2, -0.8, -1.5, -0.9, -1.1],  # Logprobs for each token
            input_tokens=[1, 2, 3, 4, 5],
            output_tokens=[[6], [7], [8], [9], [10]]
        )
        ```

        **For PMI analysis:**
        ```python
        response = ModelResponse(
            text=["The answer is 42."],
            logprobs=[-1.1, -0.9, -1.3, -0.7],  # Conditioned logprobs
            unconditioned_logprobs=[-2.1, -1.8, -2.3, -1.5],  # Unconditioned logprobs
            input_tokens=[1, 2, 3, 4],
            output_tokens=[[5], [6], [7], [8]]
        )
        ```

    Notes:
        - For most evaluation tasks, only a subset of attributes is required
        - The `text` attribute is the most commonly used for generative tasks
        - `logprobs` are essential for probability-based metrics like perplexity
        - `argmax_logits_eq_gold` is specifically for certain multiple choice/classification tasks
        - Token-level attributes (`input_tokens`, `output_tokens`) are useful for debugging
        - Truncation and padding counts help understand model behavior with long inputs
    """

    input: str | list | None = None
    text: list[str] = field(default_factory=list)  # The text of the response
    logprobs: list[float] = field(default_factory=list)  # Log probabilities of the response
    argmax_logits_eq_gold: list[bool] = field(default_factory=list)  # Whether the argmax logits match the gold text
    logits: list[list[float]] | None = None  # Logits of the response, if applicable

    truncated_tokens_count: int = 0  # How many tokens truncated
    padded_tokens_count: int = 0  # How many tokens of padding

    input_tokens: list[int] = field(default_factory=list)  # model inputs
    output_tokens: list[list[int]] = field(default_factory=list)  # model generations

    unconditioned_logprobs: Optional[list[float]] = (
        None  # Log probabilities of the unconditioned model (if applicable)
    )


@dataclass
class Batch:
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    input_lengths: list[int]
    truncated: list[int]
    padded: list[int]

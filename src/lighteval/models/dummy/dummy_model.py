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

# inspired by https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/dummy.py

import random

from transformers.models.auto.tokenization_auto import AutoTokenizer

from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig
from lighteval.tasks.requests import Doc


class DummyModelConfig(ModelConfig):
    """
    Configuration class for dummy models used for testing and baselines.

    This configuration is used to create dummy models that generate random responses
    or baselines for evaluation purposes. Useful for testing evaluation pipelines
    without requiring actual model inference.

    Attributes:
        seed (int):
            Random seed for reproducible dummy responses. Defaults to 42.
            This seed controls the randomness of the generated responses and log probabilities.

    Example:
        ```python
        config = DummyModelConfig(
            seed=123,
        )
        ```
    """

    seed: int = 42


class DummyModel(LightevalModel):
    """Dummy model to generate random baselines."""

    def __init__(
        self,
        config: DummyModelConfig,
    ):
        self.config = config
        self._random = random.Random(self.config.seed)
        self._tokenizer = None
        self.model_info = ModelInfo(model_name="dummy", model_sha=str(config.seed))

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self) -> int:
        return 2048

    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        return [ModelResponse(text=["random baseline"]) for _ in range(len(docs))]

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        model_responses = []
        for doc in docs:
            model_responses.append(
                ModelResponse(
                    logprobs=[-self._random.random() for _ in doc.choices],
                    argmax_logits_eq_gold=[False for _ in doc.choices],
                )
            )

        return model_responses

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        model_responses = []
        for doc in docs:
            model_responses.append(
                ModelResponse(
                    logprobs=[-self._random.random() for _ in doc.choices],
                    argmax_logits_eq_gold=[False for _ in doc.choices],
                )
            )

        return model_responses

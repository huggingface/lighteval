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
from typing import Optional

from transformers import AutoTokenizer

from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_config import DummyModelConfig
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse, LoglikelihoodSingleTokenResponse
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.utils import EnvConfig


class DummyModel(LightevalModel):
    """Dummy model to generate random baselines."""

    def __init__(
        self,
        config: DummyModelConfig,
        env_config: EnvConfig,
    ):
        self.config = config
        self.env_config = env_config
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

    def greedy_until(
        self, requests: list[GreedyUntilRequest], override_bs: Optional[int] = None
    ) -> list[GenerativeResponse]:
        return [GenerativeResponse(result="random baseline") for _ in range(len(requests))]

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        return [LoglikelihoodResponse((-self._random.random(), False)) for _ in requests]

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        return [LoglikelihoodResponse((-self._random.random(), False)) for _ in requests]

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        return [
            LoglikelihoodSingleTokenResponse(result=[-self._random.random() for _ in req.tokenized_continuation])
            for req in requests
        ]

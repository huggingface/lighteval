# MIT License
import random
from typing import Optional

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

# inspired by https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/dummy.py

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_config import EnvConfig
from lighteval.models.model_output import LoglikelihoodSingleTokenReturn, LoglikelihoodReturn, GenerateReturn
from lighteval.tasks.requests import LoglikelihoodSingleTokenRequest, LoglikelihoodRollingRequest, LoglikelihoodRequest, \
    GreedyUntilRequest


class DummyModel(LightevalModel):
    """Dummy model to generate random baselines."""

    def __init__(
            self,
            config,
            env_config: EnvConfig,
    ):
        self.config = config
        self.env_config = env_config

    @property
    def tokenizer(self):
        return NotImplemented

    @property
    def add_special_tokens(self):
        return NotImplemented

    @property
    def max_length(self) -> int:
        return NotImplemented

    def greedy_until(self, requests: list[GreedyUntilRequest], override_bs: Optional[int] = None) -> list[
        GenerateReturn]:
        return [GenerateReturn(result="random baseline") for _ in range(len(requests))]

    def loglikelihood(self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None) -> list[
        LoglikelihoodReturn]:
        return [LoglikelihoodReturn(-random.random()) for _ in requests]

    def loglikelihood_rolling(self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None) -> \
            list[LoglikelihoodReturn]:
        return [LoglikelihoodReturn(-random.random()) for _ in requests]

    def loglikelihood_single_token(self, requests: list[LoglikelihoodSingleTokenRequest],
                                   override_bs: Optional[int] = None) -> list[LoglikelihoodSingleTokenReturn]:
        return [
            LoglikelihoodSingleTokenReturn(result=[-random.random() for _ in req.tokenized_continuation])
            for req in requests
        ]

    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None):
        return [1]

    def tok_encode_pair(self, context, continuation):
        return [1], [2]

    def tok_decode(self, tokens) -> list[str]:
        return ["random baseline"]

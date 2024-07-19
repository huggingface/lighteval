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
from tqdm import tqdm

from transformers import AutoTokenizer
from lighteval.logging.hierarchical_logger import hlog_warn

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_config import DummyModelConfig, EnvConfig
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)


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

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self) -> int:
        return 2048

    def greedy_until(
        self, requests: list[GreedyUntilRequest], override_bs: Optional[int] = None
    ) -> list[GenerateReturn]:
        # return "random baseline" for each request
        return [GenerateReturn(result="random baseline") for _ in tqdm(range(len(requests)))]

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        # return the sum of logprobs for the n tokens in each request
        # in practice, generate a random number and multiply it by the number of tokens (this is tokenizer dependent)
        tokenized_choices = [self.tok_encode(req.choice) for req in tqdm(requests)]
        return [
            LoglikelihoodReturn(
                result=(-self._random.random() * len(tokenized_choices[i]), False),
                generated_tokens=tokenized_choices[i],
            )
            for i in range(len(requests))
        ]

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        # same as loglikelihood, but we evaluate "context" (there is no continuation, just the original sequence)
        return [
            LoglikelihoodReturn((-self._random.random() * len(self.tok_encode(req.context)), False))
            for req in tqdm(requests)
        ]

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenReturn]:
        return [
            # return one logprob per possible choice
            LoglikelihoodSingleTokenReturn(result=[-self._random.random() for _ in req.choices])
            for req in tqdm(requests)
        ]

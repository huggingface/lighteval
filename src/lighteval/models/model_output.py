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
from typing import Optional, Union

import torch


@dataclass
class ModelResponse:
    result: Union[tuple, list, str]
    input_tokens: list[int] = field(default_factory=list)  # model inputs
    generated_tokens: list[int] = field(default_factory=list)  # model generations
    truncated_tokens_count: Optional[int] = 0  # How many tokens truncated
    padded_tokens_count: Optional[int] = 0  # How many tokens of padding

    def get_result_for_eval(self):
        raise NotImplementedError()


@dataclass
class LoglikelihoodResponse(ModelResponse):
    # Float: Total log prob of the continuation
    # Optional(Bool): Whether the continuation is greedy (= all the tokens in the continuation are argmax of prob)
    result: Union[tuple[float, bool], float] = field(default_factory=tuple[float, bool])

    def get_result_for_eval(self):
        return self.result


@dataclass
class LoglikelihoodSingleTokenResponse(ModelResponse):
    # Log probs of the various single token options
    result: list[float] = field(default_factory=list)

    def get_result_for_eval(self):
        return self.result


@dataclass
class GenerativeResponse(ModelResponse):
    result: list[str] = field(default_factory=str)  # generated text continuation
    logits: Optional[list[float]] = None  # Generated text logits

    def get_result_for_eval(self):
        return self.result


@dataclass
class GenerativeMultiturnResponse(ModelResponse):
    result: list[str] = field(default_factory=list)

    def get_result_for_eval(self):
        return self.result


@dataclass
class Batch:
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    input_lengths: list[int]
    truncated: list[int]
    padded: list[int]

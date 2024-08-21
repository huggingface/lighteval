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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import AutoTokenizer, BatchEncoding

from lighteval.models.model_output import (
    GenerativeMultiturnResponse,
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.tasks.requests import (
    GreedyUntilMultiTurnRequest,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    RequestType,
)
from lighteval.utils.utils import EnvConfig


TokenSequence = Union[list[int], torch.LongTensor, torch.Tensor, BatchEncoding]


@dataclass
class ModelInfo:
    model_name: str
    model_sha: Optional[str] = None
    model_dtype: Optional[str] = None
    model_size: Optional[str] = None


class LightevalModel(ABC):
    DATASET_SPLITS = 4

    """Abstract model class defining the API that every model to plug into lighteval must follow."""

    @abstractmethod
    def __init__(
        self,
        config,
        env_config: EnvConfig,
    ):
        return NotImplemented

    def cleanup(self):
        """Clean up operations if needed, such as closing an endpoint."""
        return

    @property
    @abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        raise NotImplementedError

    @property
    @abstractmethod
    def add_special_tokens(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        raise NotImplementedError

    @property
    def disable_tqdm(self) -> bool:
        raise NotImplementedError

    def get_method_from_request_type(self, request_type: RequestType):
        if request_type == RequestType.LOGLIKELIHOOD:
            return self.loglikelihood
        if request_type == RequestType.LOGLIKELIHOOD_SINGLE_TOKEN:
            return self.loglikelihood_single_token
        if request_type == RequestType.LOGLIKELIHOOD_ROLLING:
            return self.loglikelihood_rolling
        if request_type == RequestType.GREEDY_UNTIL:
            return self.greedy_until
        if request_type == RequestType.GREEDY_UNTIL_MULTI_TURN:
            return self.greedy_until_multi_turn
        raise NotImplementedError(f"Request type {request_type} not supported")

    def greedy_until_multi_turn(  # noqa: C901
        self, requests: list[GreedyUntilMultiTurnRequest], override_bs: Optional[int] = None
    ) -> GenerativeMultiturnResponse:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met."""
        return NotImplemented

    @abstractmethod
    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            disable_tqdm (bool, optional): Whether to disable the progress bar. Defaults to False.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        return NotImplemented

    @abstractmethod
    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        return NotImplemented

    @abstractmethod
    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        return NotImplemented

    @abstractmethod
    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        return NotImplemented

    # Tokenization utils
    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None) -> TokenSequence:
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        if isinstance(str_to_encode, str):
            return self.tokenizer.encode(str_to_encode, add_special_tokens=add_special_tokens)
        return self.tokenizer(
            str_to_encode,
            padding=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

    def tok_encode_pair(self, context, continuation):
        """Encodes a context, continuation pair by taking care of the spaces in between."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def tok_decode(self, tokens: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

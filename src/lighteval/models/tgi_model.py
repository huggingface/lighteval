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

import asyncio
import math
from typing import Coroutine, List, Optional, Tuple, Union

from lighteval.data import GenerativeTaskDataset
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn
from lighteval.tasks.requests import GreedyUntilRequest, LoglikelihoodRollingRequest, LoglikelihoodSingleTokenRequest
import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from huggingface_hub import TextGenerationOutput

from lighteval.models.abstract_model import LightevalModel
from lighteval.utils import NO_TGI_ERROR_MSG, as_list, is_tgi_available


if is_tgi_available():
    from text_generation import AsyncClient


BATCH_SIZE = 50


def divide_chunks(array, n):
    # looping till length array
    for i in range(0, len(array), n):
        yield array[i : i + n]


class ModelClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(
        self,
        address,
        auth_token=None,
    ) -> None:
        if not is_tgi_available():
            raise ImportError(NO_TGI_ERROR_MSG)
        headers = {} if auth_token is None else {"Authorization": f"Basic {auth_token}"}

        self.client = AsyncClient(address, headers=headers, timeout=240)
        self._max_gen_toks = 256
        self.model_info = requests.get(f"{address}/info").json()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_id"])
        self._add_special_tokens = True

    def __process_request_generate(self, request: GreedyUntilRequest) -> TextGenerationOutput:
        
        stop_sequences = as_list(request.stop_sequence)
        
        if stop_sequences is None or stop_sequences == [None]:
            stop_sequences = []

        generated_text = self.client.generate(
            request.context,
            max_new_tokens=request.generation_size,
            decoder_input_details=False,
            stop_sequences=stop_sequences,
            seed=42,
            truncate=ModelClient._DEFAULT_MAX_LENGTH,
        )

        return generated_text

    async def __process_batch_generate(self, requests: List[GreedyUntilRequest]):
        return await asyncio.gather(*[self.__process_request_generate(request) for request in requests])

    def greedy_until(self, requests: List[GreedyUntilRequest], override_bs=None) -> List[str]:
        batch_size = override_bs if override_bs > 0 else BATCH_SIZE

        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]

        dataset = GenerativeTaskDataset(requests=requests, dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[GenerateReturn] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                # the `returns_logits` flag is only used to filter the results, we always request the full details.
                returns_logits = batch[0].use_logits
                num_samples = batch[0].num_samples
                if num_samples > 1:
                    raise Exception(
                        "Inference endpoints does not allow sampling evaluations - this is likely to fail or provide problematic results"
                    )

                responses = asyncio.run(self.__process_batch_generate(batch))
                for response in responses:
                    results.append(
                        GenerateReturn(
                            result=response.generated_text,
                            logits=[item.logprob for item in response.details.prefill] if returns_logits else None,
                            truncated_tokens_count=-1,
                            padded_tokens_count=-1,
                        )
                    )

        return dataset.get_original_order(results)


    def __process_request_logprob(self, request: Tuple[str, str]) -> Coroutine[None, List, str]:
        context, choice = request
        out = self.client.generate(context + choice, max_new_tokens=1, decoder_input_details=True)
        return out

    async def __process_batch_logprob(self, requests: List[Tuple[str, str]]):
        return await asyncio.gather(*[self.__process_request_logprob(request) for request in requests])

    def loglikelihood(self, requests: List[Tuple[str, str]], override_bs=None) -> List[Tuple[float, bool]]:
        res: List[Tuple[float, bool]] = []

        batch_size = override_bs if override_bs > 0 else BATCH_SIZE

        for batch in tqdm(
            divide_chunks(requests, batch_size), total=math.ceil(len(requests) // batch_size), maxinterval=1
        ):
            results = asyncio.run(self.__process_batch_logprob(batch))
            details = [result.details.prefill for result in results]

            for detail, (context, choice) in zip(details, batch):
                tokenized_context = self.tokenizer.tokenize(context, add_special_tokens=self._add_special_tokens)
                tokenized_input = self.tokenizer.tokenize(context + choice, add_special_tokens=self._add_special_tokens)

                i = 0
                while i < len(tokenized_context) and tokenized_input[i] == tokenized_context[i]:
                    i += 1

                logprobs = [token.logprob for token in detail[i:]]

                logit_sum: float = np.sum(logprobs)
                res.append((logit_sum, False))

        return res

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        return NotImplemented

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenReturn]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        return NotImplemented

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return ModelClient._DEFAULT_MAX_LENGTH

    @property
    def disable_tqdm(self) -> bool:
        False

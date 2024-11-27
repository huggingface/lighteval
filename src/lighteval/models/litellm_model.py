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

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset
from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoint_model import ModelInfo
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.imports import is_litellm_available


if is_litellm_available():
    import logging

    import litellm
    from litellm.caching.caching import Cache

    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    litellm.cache = Cache(type="disk")


class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config, env_config) -> None:
        """
        IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )
        self.provider = config.provider
        self.base_url = os.getenv(f"{config.provider.upper()}_BASE_URL", None)
        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.CONCURENT_CALLS = 20  # 100 leads to hitting Anthropic rate limits
        self.TEMPERATURE = 0.7
        self.TOP_P = 0.95
        self.model = config.model
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use a dummy tokenizer for compatibility
        self.pairwise_tokenization = False
        litellm.drop_params = True

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence, generation_size):
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens if max_new_tokens > 0 else None,
                    logprobs=return_logits if self.provider == "openai" else None,
                    base_url=self.base_url,
                    n=num_samples,
                    temperature=self.TEMPERATURE,
                    top_p=self.TOP_P,
                    stop=["\n"] if stop_sequence is None else stop_sequence,
                    max_completion_tokens=generation_size if generation_size > 0 else None,
                    caching=True,
                )
                return response
            except litellm.exceptions.RateLimitError:
                if attempt == self.API_MAX_RETRY - 1:
                    raise
                wait_time = min(64, self.API_RETRY_SLEEP * (2**attempt))  # Exponential backoff with max 64s
                hlog_warn(
                    f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)
            except Exception as e:
                hlog_warn(f"{type(e), e}")
                if attempt == self.API_MAX_RETRY - 1:
                    raise
                wait_time = self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                hlog_warn(f"Retrying in {wait_time} seconds")
                time.sleep(wait_time)

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int],
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
        generation_size: int | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        generation_sizess = [generation_size for _ in prompts]

        assert (
            len(prompts)
            == len(return_logitss)
            == len(max_new_tokenss)
            == len(num_sampless)
            == len(stop_sequencess)
            == len(generation_sizess)
        ), f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, generation_sizes should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}, {len(generation_sizess)}"

        with ThreadPoolExecutor(self.CONCURENT_CALLS) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                    generation_sizess,
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

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
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            contexts = [c.context for c in dataset]
            max_new_tokens = dataset[0].generation_size  # could be none
            return_logits = dataset[0].use_logits
            num_samples = dataset[0].num_samples
            stop_sequence = requests[0].stop_sequence
            generation_size = requests[0].generation_size

            responses = self.__call_api_parallel(
                contexts, return_logits, max_new_tokens, num_samples, stop_sequence, generation_size
            )

            for response in responses:
                result: list[str] = [choice.message.content for choice in response.choices]

                cur_response = GenerativeResponse(
                    result=result,
                    logits=None,
                    generated_tokens=[],
                    input_tokens=[],
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    def tok_encode(self, text: str):
        return self.tokenizer.encode(text)

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

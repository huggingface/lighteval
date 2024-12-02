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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import anthropic
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
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
from lighteval.utils.imports import is_openai_available


logger = logging.getLogger(__name__)

if is_openai_available():
    pass

    # logging.getLogger("openai").setLevel(logging.ERROR)
    # logging.getLogger("httpx").setLevel(logging.ERROR)


class AnthropicClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config, env_config) -> None:
        # api_key = os.environ["ANTHROPIC_API_KEY"]
        self.client = anthropic.Anthropic()

        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )
        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.CONCURENT_CALLS = 100
        self.model = config.model
        # self._tokenizer = tiktoken.encoding_for_model(self.model)
        self.pairwise_tokenization = False

    def __call_api(self, prompt, max_new_tokens, num_samples):
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens if max_new_tokens > 0 else None,
                    n=num_samples,
                )
                return response
            except Exception as e:
                logger.warning(f"{type(e), e}")
                time.sleep(self.API_RETRY_SLEEP)
                self.API_RETRY_SLEEP = self.API_RETRY_SLEEP**self.API_RETRY_MULTIPLIER
        raise Exception("Failed to get response from the API")

    def __call_api_parallel(
        self,
        prompts,
        max_new_tokens: int | list[int],
        num_samples: int | list[int],
    ):
        results = []

        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples

        assert (
            len(prompts) == len(max_new_tokenss) == len(num_sampless)
        ), "Length of prompts, max_new_tokenss, num_sampless, should be the same"

        with ThreadPoolExecutor(self.CONCURENT_CALLS) as executor:
            for entry in tqdm(
                executor.map(self.__call_api, prompts, max_new_tokenss, num_sampless),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in __call_api, please inspect and retry.")

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
            max_new_tokens = dataset[0].generation_size  # could be none
            num_samples = dataset[0].num_samples
            contexts = [c.context for c in dataset]

            responses = self.__call_api_parallel(contexts, max_new_tokens, num_samples)

            for response in responses:
                result: list[str] = [output.content for output in response]

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

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
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

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
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

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
from lighteval.utils.imports import is_openai_available


if is_openai_available():
    import logging

    import tiktoken
    from openai import OpenAI

    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)


API_MAX_RETRY = 5
API_RETRY_SLEEP = 3
API_RETRY_MULTIPLIER = 2
CONCURENT_CALLS = 100


class OpenAIClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config, env_config) -> None:
        api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key)

        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )
        self.model = config.model
        self._tokenizer = tiktoken.encoding_for_model(self.model)
        self.pairwise_tokenization = False

    @retry(
        stop=stop_after_attempt(API_MAX_RETRY),
        wait=wait_random_exponential(multiplier=API_RETRY_SLEEP, exp_base=API_RETRY_MULTIPLIER),
        before_sleep=lambda retry_state: hlog_warn(
            f"API call failed, retrying... ({retry_state.attempt_number}/{API_MAX_RETRY})"
        ),
        reraise=True,
    )
    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, logit_bias):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "text"},
                max_tokens=max_new_tokens if max_new_tokens > 0 else None,
                logprobs=return_logits,
                logit_bias=logit_bias,
                n=num_samples,
            )
            return response
        except Exception as e:
            hlog_warn(f"API call failed: {type(e).__name__}: {str(e)}")
            raise

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int],
        num_samples: int | list[int],
        logit_bias: list[dict[int, float]] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        logit_biass = [logit_bias for _ in prompts] if logit_bias is None else logit_bias

        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(logit_biass)
        ), "Length of prompts, return_logitss, max_new_tokenss, num_sampless, logit_biass should be same"

        with ThreadPoolExecutor(CONCURENT_CALLS) as executor:
            for entry in tqdm(
                executor.map(self.__call_api, prompts, return_logitss, max_new_tokenss, num_sampless, logit_biass),
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
            max_new_tokens = dataset[0].generation_size  # could be none
            return_logits = dataset[0].use_logits
            num_samples = dataset[0].num_samples
            contexts = [c.context for c in dataset]

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples)

            for response in responses:
                result: list[str] = [output.message.content for output in response.choices]

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
        for request in requests:
            if request.context == "":
                request.tokenized_context = [" "]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice, pairwise=self.pairwise_tokenization
                )
        return self._loglikelihood_tokens(requests)

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
    ) -> list[LoglikelihoodResponse]:
        original_requests = requests

        # First group the requests by context, use itertools.groupby
        from itertools import groupby

        assert all(len(req.tokenized_continuation) == 1 for req in requests)
        grouped_requests = [list(g) for k, g in groupby(requests, lambda x: x.sample_index)]
        assert all(
            all(requests[0].tokenized_context == request.tokenized_context for request in requests)
            for requests in grouped_requests
        )

        results = {}

        # Then for each group, call the API
        for i in tqdm(range(0, len(grouped_requests), 100)):
            batch = grouped_requests[i : i + 100]
            contexts = [request[0].context for request in batch]
            continuation_tokens = [[request.tokenized_continuation[0] for request in requests] for requests in batch]

            logit_biass = []
            for toks in continuation_tokens:
                logit_bias = {tok: 100 for tok in toks}
                logit_biass.append(logit_bias)

            outputs = self.__call_api_parallel(
                contexts, return_logits=True, max_new_tokens=1, num_samples=1, logit_bias=logit_biass
            )

            for requests, outs in zip(batch, outputs):
                # First find which token we got
                output_chars = outs.choices[0].message.content
                logprob = outs.choices[0].logprobs.content[0].logprob

                # Get index of the matching choice
                chosen_index = [request.choice for request in requests].index(output_chars)
                if chosen_index == -1:
                    raise ValueError("Choice not found in the list of choices")
                results[requests[0].sample_index] = {
                    request.request_index: LoglikelihoodResponse(
                        input_tokens=request.tokenized_context,
                        generated_tokens=request.tokenized_continuation,
                        result=(float("-inf") if chosen_index != i else logprob, None),
                    )
                    for i, request in enumerate(requests)
                }
        sorted_results = [results[request.sample_index][request.request_index] for request in original_requests]

        return sorted_results

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

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

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
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
        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.CONCURENT_CALLS = 100
        self.model = config.model
        self._tokenizer = tiktoken.encoding_for_model(self.model)
        self.pairwise_tokenization = False

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, logit_bias):
        for _ in range(self.API_MAX_RETRY):
            try:
                print(prompt)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "text"},
                    max_tokens=max_new_tokens if max_new_tokens > 0 else 100,
                    logprobs=return_logits,
                    top_logprobs=3,
                    logit_bias=logit_bias,
                    n=num_samples,
                )
                return response
            except Exception as e:
                hlog_warn(f"{type(e), e}")
                time.sleep(self.API_RETRY_SLEEP)
        raise Exception("Failed to get response from the API")

    def __call_api_parallel(self, prompts, return_logits, max_new_tokens, num_samples, logit_bias=None):
        results = []
        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        logit_biass = [logit_bias for _ in prompts] if logit_bias is None else logit_bias

        with ThreadPoolExecutor(self.CONCURENT_CALLS) as executor:
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
            # pprint(responses)

            for response in responses:
                result: list[str] = [output.message.content for output in response.choices]
                # pprint(result)

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
                    request.context, request.choice.strip(), pairwise=self.pairwise_tokenization
                )
        return self._loglikelihood_tokens(requests)

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
    ) -> list[LoglikelihoodResponse]:
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=1)
        results = []

        for _ in tqdm(dataset.splits_start_end_iterator()):
            inputs = [dataset[i].context for i in range(len(dataset))]
            logit_biass = []
            max_new_tokens = [len(dataset[i].tokenized_continuation) for i in range(len(dataset))]

            for i in range(len(dataset)):
                logit_bias = {tok: 100 for tok in dataset[i].tokenized_continuation}
                logit_biass.append(logit_bias)

            from pprint import pprint

            outputs = self.__call_api_parallel(
                inputs, return_logits=True, max_new_tokens=max_new_tokens, num_samples=1, logit_bias=logit_biass
            )

            # pprint(outputs)

            for output, input in zip(outputs, dataset):
                pprint(output.choices[0].logprobs)
                pprint(output.choices[0].message)
                pprint(input.choice)
                pprint(input.tokenized_continuation)
                print("=======")

                # continuation_logprobs = []

                # for token, logprobs in zip(input.tokenized_continuation[::-1], output.prompt_logprobs[::-1]):
                #    continuation_logprobs.append(logprobs[token])

                # bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                # continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
                # answer = LoglikelihoodResponse(
                #    input_tokens=input.tokenized_context + input.tokenized_continuation,
                #    generated_tokens=input.tokenized_continuation,
                #    result=(sum(continuation_logprobs), bool_score if return_bool_score else None),
                # )
                # res.append(answer)

        return dataset.get_original_order(results)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        return NotImplemented

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        return NotImplemented

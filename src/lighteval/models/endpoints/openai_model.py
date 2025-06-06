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
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_input import GenerationParameters
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
    import logging

    import tiktoken
    from openai import OpenAI

    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)


@dataclass
class OpenAIModelConfig:
    model: str
    generation_parameters: GenerationParameters = None
    base_url: str = "https://api.openai.com/v1"
    api_key: str = os.environ.get("OPENAI_API_KEY", None)

    def __post_init__(self):
        if not self.generation_parameters:
            self.generation_parameters = GenerationParameters()

    @classmethod
    def from_path(cls, path: str) -> "OpenAIModelConfig":
        import yaml

        with open(path, "r") as f:
            loaded_file = yaml.safe_load(f)
            config = loaded_file["model"]
            api = loaded_file.get("api", {})
        generation_parameters = GenerationParameters.from_dict(config)
        return cls(model=config["model_name"], generation_parameters=generation_parameters, **api)


class OpenAIClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: OpenAIModelConfig, env_config) -> None:
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.config = config
        self.generation_parameters = config.generation_parameters
        self.sampling_params = self.generation_parameters.to_vllm_openai_dict()

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
        try:
            self._tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pairwise_tokenization = False

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, logit_bias):
        for _ in range(self.API_MAX_RETRY):
            try:
                response_format = {"response_format": {"type": "text"}} if "openai" in self.config.base_url else {}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens if max_new_tokens > 0 else None,
                    logprobs=return_logits,
                    logit_bias=logit_bias,
                    n=num_samples,
                    **self.sampling_params,
                    **response_format,
                )
                self.API_RETRY_SLEEP = 3
                return response
            except Exception as e:
                logger.warning(f"{type(e), e}")
                time.sleep(self.API_RETRY_SLEEP)
                self.API_RETRY_SLEEP = self.API_RETRY_SLEEP**self.API_RETRY_MULTIPLIER
        raise Exception("Failed to get response from the API")

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

        assert len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(logit_biass), (
            "Length of prompts, return_logitss, max_new_tokenss, num_sampless, logit_biass should be same"
        )

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
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            contexts = [sample.context for sample in split]

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
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=1)
        results = []

        for split in tqdm(dataset.splits_iterator()):
            inputs = [sample.context for sample in split]
            max_new_tokens = [len(sample.tokenized_continuation) for sample in split]

            assert all(new_tokens == 1 for new_tokens in max_new_tokens), (
                "Only single token continuations are supported when using openai API."
            )

            logit_biases = [dict.fromkeys(sample.tokenized_continuation, 100) for sample in split]

            outputs = self.__call_api_parallel(
                inputs, return_logits=True, max_new_tokens=max_new_tokens, num_samples=1, logit_bias=logit_biases
            )

            for i, output in enumerate(outputs):
                input = split[i]
                continuation_logprobs = [content.logprob for content in output.choices[0].logprobs.content]
                answer = LoglikelihoodResponse(
                    input_tokens=input.tokenized_context + input.tokenized_continuation,
                    generated_tokens=input.tokenized_continuation,
                    result=(sum(continuation_logprobs), None),
                )
                results.append(answer)

        return dataset.get_original_order(results)

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

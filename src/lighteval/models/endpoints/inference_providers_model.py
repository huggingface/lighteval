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
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml
from huggingface_hub import AsyncInferenceClient, ChatCompletionOutput
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset
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


logger = logging.getLogger(__name__)


@dataclass
class InferenceProvidersModelConfig:
    model: str
    provider: str
    timeout: int | None = None
    proxies: Any | None = None
    generation_parameters: GenerationParameters = field(default_factory=GenerationParameters)

    @classmethod
    def from_path(cls, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)["model"]

        model = config["model_name"]
        provider = config.get("provider", None)
        timeout = config.get("timeout", None)
        proxies = config.get("proxies", None)
        generation_parameters = GenerationParameters.from_dict(config)
        return cls(
            model=model,
            provider=provider,
            timeout=timeout,
            proxies=proxies,
            generation_parameters=generation_parameters,
        )


class InferenceProvidersClient(LightevalModel):
    def __init__(self, config) -> None:
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
        self.model = config.model
        self.provider = config.provider
        self.generation_parameters = config.generation_parameters

        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.pairwise_tokenization = False

        self.client = AsyncInferenceClient(
            provider=self.provider,
            timeout=config.timeout,
            proxies=config.proxies,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

    def _encode(self, text: str):
        enc = self._tokenizer(text=text)
        return enc

    def tok_encode(self, text: str | list[str]):
        if isinstance(text, list):
            toks = [self._encode(t["content"]) for t in text]
            toks = [tok for tok in toks if tok]
            return toks
        return self._encode(text)

    async def __call_api(self, prompt, num_samples):
        """Make API call with retries."""
        for attempt in range(self.API_MAX_RETRY):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": prompt,
                    "n": num_samples,
                }
                kwargs.update(self.generation_parameters.to_inference_providers_dict())
                response: ChatCompletionOutput = await self.client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                wait_time = min(64, self.API_RETRY_SLEEP * (2**attempt))  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        return None

    async def __call_api_parallel(
        self,
        prompts,
        num_samples: int | list[int],
    ):
        results = []

        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        assert (
            len(prompts) == len(num_sampless)
        ), f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(num_sampless)}"

        tasks = [self.__call_api(prompt, num_samples) for prompt, num_samples in zip(prompts, num_sampless)]
        results = await async_tqdm.gather(*tasks)

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

        for _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            contexts = [c.context for c in dataset]
            num_samples = dataset[0].num_samples

            responses = asyncio.run(self.__call_api_parallel(contexts, num_samples))

            for response in responses:
                result: list[str] = [choice.message.content for choice in response.choices]

                cur_response = GenerativeResponse(
                    # In empty responses, the model should return an empty string instead of None
                    result=result if result[0] else [""],
                    logits=None,
                    generated_tokens=[],
                    input_tokens=[],
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

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

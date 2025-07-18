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
from typing import Any, List, Optional

from huggingface_hub import AsyncInferenceClient, ChatCompletionOutput
from huggingface_hub.errors import HfHubHTTPError
from pydantic import NonNegativeInt
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


class InferenceProvidersModelConfig(ModelConfig):
    """
    Configuration class for HuggingFace's inference providers (like Together AI, Anyscale, etc.).

    inference providers doc: https://huggingface.co/docs/inference-providers/en/index

    Attributes:
        model_name (str):
            Name or identifier of the model to use.
        provider (str):
            Name of the inference provider. Examples: "together", "anyscale", "runpod", etc.
        timeout (int | None):
            Request timeout in seconds. If None, uses provider default.
        proxies (Any | None):
            Proxy configuration for requests. Can be a dict or proxy URL string.
        org_to_bill (str | None):
            Organization to bill for API usage. If None, bills the user's account.
        parallel_calls_count (NonNegativeInt):
            Number of parallel API calls to make. Defaults to 10.
            Higher values increase throughput but may hit rate limits.

    Example:
        ```python
        config = InferenceProvidersModelConfig(
            model_name="deepseek-ai/DeepSeek-R1-0528",
            provider="together",
            parallel_calls_count=5,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```

    Note:
        - Requires HF API keys to be set in environment variable
        - Different providers have different rate limits and pricing
    """

    model_name: str
    provider: str
    timeout: int | None = None
    proxies: Any | None = None
    org_to_bill: str | None = None
    parallel_calls_count: NonNegativeInt = 10


class InferenceProvidersClient(LightevalModel):
    """Client for making inference requests to various providers using the HuggingFace Inference API.

    This class handles batched generation requests with automatic retries and error handling.
    API keys should be set in environment variables.
    """

    def __init__(self, config: InferenceProvidersModelConfig) -> None:
        """Initialize the inference client.

        Args:
            config: Configuration object containing model and provider settings
        """
        self.model_info = ModelInfo(
            model_name=config.model_name,
            model_sha="",
            model_dtype=None,
            model_size=-1,
        )
        self.model_name = config.model_name
        self.provider = config.provider
        self.generation_parameters = config.generation_parameters

        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.pairwise_tokenization = False
        self.semaphore = asyncio.Semaphore(config.parallel_calls_count)  # Limit concurrent API calls

        self.client = AsyncInferenceClient(
            provider=self.provider,
            timeout=config.timeout,
            proxies=config.proxies,
            bill_to=config.org_to_bill,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except HfHubHTTPError:
            logger.warning(f"Could not load model's tokenizer for the model {self.model_name}.")
            self._tokenizer = None

        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

    async def __call_api(self, prompt: List[dict], num_samples: int) -> Optional[ChatCompletionOutput]:
        """Make API call with exponential backoff retry logic.

        Args:
            prompt: List of message dictionaries for chat completion
            num_samples: Number of completions to generate

        Returns:
            API response or None if all retries failed
        """
        for attempt in range(self.API_MAX_RETRY):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": prompt,
                    "n": num_samples,
                }
                kwargs.update(self.generation_parameters.to_inference_providers_dict())
                if kwargs.get("temperature") == 0.0 and num_samples > 1:
                    raise ValueError(
                        "Temperature is set to 0.0, but num_samples > 1. "
                        "This is not supported by the inference providers API."
                    )
                response: ChatCompletionOutput = await self.client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                wait_time = min(64, self.API_RETRY_SLEEP * (2**attempt))  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                await asyncio.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        return None

    async def __call_api_parallel(
        self,
        prompts,
        num_samples: int | list[int],
    ):
        results = []

        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        assert len(prompts) == len(num_sampless), (
            f"Length of prompts and max_new_tokenss should be the same but are {len(prompts)}, {len(num_sampless)}"
        )

        async def bounded_api_call(prompt, num_samples):
            async with self.semaphore:
                return await self.__call_api(prompt, num_samples)

        tasks = [bounded_api_call(prompt, num_samples) for prompt, num_samples in zip(prompts, num_sampless)]
        results = await async_tqdm.gather(*tasks)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in __call_api, please inspect and retry.")

        return results

    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in split]
            num_samples = split[0].num_samples

            responses = asyncio.run(self.__call_api_parallel(contexts, num_samples))

            for response, context in zip(responses, contexts):
                result: list[str] = [choice.message.content for choice in response.choices]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result[0] else [""],
                    input=context,
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
        try:
            return self._tokenizer.model_max_length
        except AttributeError:
            logger.warning("Tokenizer was not correctly loaded. Max model context length is assumed to be 30K tokens")
            return 30000

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError

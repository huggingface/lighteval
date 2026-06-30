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
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError

import requests
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import is_package_available, requires


logger = logging.getLogger(__name__)

if is_package_available("litellm"):
    import litellm
    from litellm import encode, supports_reasoning
    from litellm.caching.caching import Cache, LiteLLMCacheType
    from litellm.utils import ModelResponse as LitellmModelResponse
    from litellm.utils import get_max_tokens

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    litellm.cache = Cache(type=LiteLLMCacheType.DISK)
else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()
    LitellmModelResponse = Mock()


class LiteLLMModelConfig(ModelConfig):
    """Configuration class for LiteLLM unified API client.

    This configuration is used to connect to various LLM providers through the LiteLLM
    unified API. LiteLLM provides a consistent interface to multiple providers including
    OpenAI, Anthropic, Google, and many others.

    litellm doc: https://docs.litellm.ai/docs/

    Attributes:
        model_name (str):
            Model identifier. Can include provider prefix (e.g., "gpt-4", "claude-3-sonnet")
            or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        provider (str | None):
            Optional provider name override. If None, inferred from model_name.
            Examples: "openai", "anthropic", "google", "cohere", etc.
        base_url (str | None):
            Custom base URL for the API. If None, uses provider's default URL.
            Useful for using custom endpoints or local deployments.
        api_key (str | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).
        concurrent_requests (int):
            Maximum number of concurrent API requests to execute in parallel.
            Higher values can improve throughput for batch processing but may hit rate limits
            or exhaust API quotas faster. Default is 10.
        verbose (bool):
            Whether to enable verbose logging. Default is False.
        max_model_length (int | None):
            Maximum context length for the model. If None, infers the model's default max length.
        api_max_retry (int):
            Maximum number of retries for API requests. Default is 8.
        api_retry_sleep (float):
            Initial sleep time (in seconds) between retries. Default is 1.0.
        api_retry_multiplier (float):
            Multiplier for increasing sleep time between retries. Default is 2.0.
        timeout (float):
            Request timeout in seconds. Default is None (no timeout).
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.

    Supported evaluation modes:
        - ``greedy_until`` (generative): all models and providers supported.
        - ``loglikelihood`` (MCQ ranking) and ``loglikelihood_rolling`` (perplexity):
          requires the ``/v1/completions`` endpoint with ``echo=True`` and
          ``logprobs=1``.  Supported by ``gpt-3.5-turbo-instruct`` and any
          OpenAI-compatible local server (llama.cpp, vLLM serve, etc.).
          Chat-only models (gpt-4o, Claude, Gemini) are **not** supported for
          these modes; the backend will warn at eval start and return ``-inf``.

    Example:
        ```python
        # Generative tasks only (any provider)
        config = LiteLLMModelConfig(
            model_name="gpt-4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            concurrent_requests=5,
            generation_parameters=GenerationParameters(temperature=0.7, max_new_tokens=100),
        )

        # MCQ / perplexity tasks (requires /v1/completions support)
        config = LiteLLMModelConfig(
            model_name="gpt-3.5-turbo-instruct",
            provider="openai",
            concurrent_requests=10,
            generation_parameters=GenerationParameters(seed=42),
        )
        ```
    """

    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    concurrent_requests: int = 10
    verbose: bool = False
    max_model_length: int | None = None

    api_max_retry: int = 8
    api_retry_sleep: float = 1.0
    api_retry_multiplier: float = 2.0
    timeout: float | None = None


@requires("litellm")
class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: LiteLLMModelConfig) -> None:
        """IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.config = config
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self._max_length = config.max_model_length

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier
        self.timeout = config.timeout

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.verbose = config.verbose
        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens) -> int | None:
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if supports_reasoning(self.model):
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, self.max_length)

            logger.warning(
                f"Reasoning model detected, increasing max_new_tokens to {max_new_tokens} to allow for reasoning tokens",
            )

        return max_new_tokens

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):  # noqa: C901
        """Make API call with retries."""
        response = LitellmModelResponse()
        stop_sequence = self._prepare_stop_sequence(stop_sequence)
        max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

        if return_logits and not self.provider == "openai":
            logger.warning("Returning logits is not supported for this provider, ignoring.")

        # Prepare kwargs for completion call
        kwargs = {
            "model": self.model,
            "messages": prompt,
            "response_format": {"type": "text"},
            "max_tokens": max_new_tokens,
            "logprobs": return_logits if self.provider == "openai" else None,
            "stop": stop_sequence,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "n": num_samples,
            "caching": True,
            "timeout": self.timeout,
        }

        if "o1" in self.model:
            logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
        else:
            kwargs.update(self.generation_parameters.to_litellm_dict())

        if kwargs.get("max_completion_tokens", None) is None:
            kwargs["max_completion_tokens"] = max_new_tokens

        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if not content:
                    logger.info("Response is empty, retrying without caching")
                    kwargs["caching"] = False
                    response = litellm.completion(**kwargs)
                    content = response.choices[0].message.content

                return response
            except litellm.BadRequestError as e:
                if "message" in e.__dict__:
                    error_string = (
                        "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    )
                    if error_string in e.__dict__["message"]:
                        logger.warning(f"{error_string}. Returning empty response.")
                        return LitellmModelResponse()
            except Exception as e:
                wait_time = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        return LitellmModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), (
            f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"
        )

        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def estimate_context_length(self) -> int:
        def fallback():
            logger.warning("Failed to fetch model endpoint info from OpenRouter, returning default max length.")
            return self._DEFAULT_MAX_LENGTH

        # If the model is used through openrouter, the actual model name comes after the prefix
        model_name = self.model.removeprefix("openrouter/")
        endpoint_info_response = requests.get(
            f"https://openrouter.ai/api/v1/models/{model_name}/endpoints",
            headers={},
        )
        if endpoint_info_response.ok:
            try:
                endpoint_info = endpoint_info_response.json()
                context_lengths = {
                    endpoint["provider_name"]: endpoint["context_length"]
                    for endpoint in endpoint_info["data"]["endpoints"]
                }

                if self.provider in context_lengths:
                    return context_lengths[self.provider]

                min_length = min(context_lengths.values())
                logger.warning(
                    f"Estimating model context length as the minimum context length from available OpenRouter providers: {min_length}"
                )
                return min_length
            except (KeyError, TypeError, ValueError, JSONDecodeError):
                return fallback()

        return fallback()

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in split]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence)

            for response, context in zip(responses, contexts):
                result: list[str] = [choice.message.content for choice in response.choices]
                reasonings: list[str | None] = [
                    getattr(choice.message, "reasoning_content", None) for choice in response.choices
                ]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result[0] else [""],
                    reasonings=reasonings,
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
        if self._max_length is not None:
            return self._max_length

        try:
            max_tokens = get_max_tokens(self.model)
        except Exception:
            logger.error(
                f"Unable to get the maximum sequence length for model {self.model} from litellm. Fetching information from OpenRouter instead."
            )
            max_tokens = self.estimate_context_length()

        # Avoid future requests
        self._max_length = max_tokens

        return max_tokens

    # ------------------------------------------------------------------
    # Token Alignment Engine helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_continuation_start(logprobs_obj, context_str: str, model: str) -> int:
        """Return the index in the token sequence where the continuation begins.

        Two-layer strategy:
          1. Character-offset alignment via ``text_offset`` (exact, uses the
             API's own tokenisation — preferred when available).
          2. Tiktoken-count fallback via ``litellm.encode`` (reliable for all
             OpenAI-family models when ``text_offset`` is absent).
        """
        # Layer 1 — character-offset alignment
        text_offset = getattr(logprobs_obj, "text_offset", None)
        if text_offset:
            context_char_len = len(context_str)
            for i, offset in enumerate(text_offset):
                if offset >= context_char_len:
                    return i
            return len(text_offset)  # empty continuation

        # Layer 2 — tiktoken fallback
        try:
            ctx_toks = encode(model, context_str)
            return len(ctx_toks)
        except Exception:
            logger.warning(
                "Could not align continuation tokens via text_offset or tiktoken. "
                "Logprob results may be inaccurate for this provider."
            )
            return 0

    @staticmethod
    def _check_argmax(
        tokens: list[str],
        token_logprobs: list,
        top_logprobs: list,
        cont_start: int,
    ) -> bool:
        """Return True if every continuation token was the model's top-1 prediction.

        Mirrors vLLM's ``rank == 1`` check. The last token in ``tokens`` is the
        newly generated token (from ``max_tokens=1``) and is excluded from the
        check.
        """
        if not top_logprobs:
            return False

        cont_end = len(tokens) - 1  # exclude the single generated token at the end
        if cont_start >= cont_end:
            return True  # empty continuation trivially matches

        for i in range(cont_start, cont_end):
            if i >= len(top_logprobs):
                return False
            top_dict = top_logprobs[i]
            if not top_dict:
                return False
            # With logprobs=1, top_dict has exactly one key: the top-1 token.
            top_token = next(iter(top_dict))
            if i >= len(tokens) or tokens[i] != top_token:
                return False

        return True

    # ------------------------------------------------------------------
    # Async loglikelihood implementation
    # ------------------------------------------------------------------

    async def _call_api_text_completion_async(
        self,
        full_text: str,
        semaphore: asyncio.Semaphore,
    ):
        """Async call to ``litellm.atext_completion`` with exponential backoff.

        Uses ``echo=True``, ``logprobs=1``, ``max_tokens=1``, and
        ``temperature=0.0`` to retrieve per-token log-probabilities for every
        token in ``full_text`` (context + continuation).

        HTTP 429 (``RateLimitError``) is handled explicitly before the generic
        exception handler so that rate-limit pauses are clearly logged.
        """
        async with semaphore:
            for attempt in range(self.API_MAX_RETRY):
                try:
                    response = await litellm.atext_completion(
                        model=self.model,
                        prompt=full_text,
                        max_tokens=1,  # generate exactly 1 token (echo gives prompt logprobs)
                        echo=True,  # return prompt tokens with their log-probabilities
                        logprobs=1,  # top-1 logprob per position for argmax check
                        temperature=0.0,  # deterministic scoring
                        base_url=self.base_url,
                        api_key=self.api_key,
                        caching=True,
                        timeout=self.timeout,
                        **self.generation_parameters.to_litellm_text_completion_dict(),
                    )
                    return response
                except litellm.RateLimitError:
                    wait_time = min(64.0, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
                    logger.warning(
                        f"Rate limit (HTTP 429) on loglikelihood call — "
                        f"backing off {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.API_MAX_RETRY})"
                    )
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    wait_time = min(64.0, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
                    logger.warning(
                        f"Error in loglikelihood API call: {e} — "
                        f"backing off {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.API_MAX_RETRY})"
                    )
                    await asyncio.sleep(wait_time)

            logger.error(
                f"Loglikelihood API call failed after {self.API_MAX_RETRY} attempts, returning None."
            )
            return None

    async def _process_doc_loglikelihood_async(
        self,
        doc: Doc,
        context_str: str,
        semaphore: asyncio.Semaphore,
    ) -> ModelResponse:
        """Compute logprobs for all choices of a single doc concurrently.

        All (context + choice) API calls for this doc are fired at once via
        ``asyncio.gather``, bounded by the shared semaphore.  Returns a
        ``ModelResponse`` with ``logprobs`` and ``argmax_logits_eq_gold``
        populated per-choice, matching the VLLMModel data contract exactly.
        """
        # Soft length check using the longest choice as a conservative estimate
        if doc.choices:
            longest = max(doc.choices, key=len)
            self._warn_if_too_long(context_str + longest, label=f"doc '{doc.id}' longest choice")

        tasks = [
            self._call_api_text_completion_async(context_str + choice, semaphore)
            for choice in doc.choices
        ]
        responses = await asyncio.gather(*tasks)

        logprobs_per_choice: list[float] = []
        argmax_per_choice: list[bool] = []

        for choice, response in zip(doc.choices, responses):
            if response is None or not getattr(response, "choices", None):
                logprobs_per_choice.append(float("-inf"))
                argmax_per_choice.append(False)
                continue

            lp_obj = getattr(response.choices[0], "logprobs", None)
            if lp_obj is None or not getattr(lp_obj, "token_logprobs", None):
                logprobs_per_choice.append(float("-inf"))
                argmax_per_choice.append(False)
                continue

            tokens: list[str] = list(lp_obj.tokens or [])
            token_logprobs: list = list(lp_obj.token_logprobs or [])
            top_logprobs: list = list(lp_obj.top_logprobs or [])

            cont_start = self._find_continuation_start(lp_obj, context_str, self.model)

            # token_logprobs[cont_start:-1] isolates the continuation slice.
            # The -1 excludes the single token generated by max_tokens=1 which
            # is appended at the very end of the echoed sequence.
            cont_lp_slice = token_logprobs[cont_start:-1]
            valid_lp = [v for v in cont_lp_slice if v is not None]
            total_logprob = sum(valid_lp) if valid_lp else float("-inf")

            is_argmax = self._check_argmax(tokens, token_logprobs, top_logprobs, cont_start)

            logprobs_per_choice.append(total_logprob)
            argmax_per_choice.append(is_argmax)

        return ModelResponse(
            input=context_str,
            logprobs=logprobs_per_choice,
            argmax_logits_eq_gold=argmax_per_choice,
        )

    async def _loglikelihood_async(self, docs: list[Doc]) -> list[ModelResponse]:
        """Async coordinator: process every doc in parallel, bounded by the semaphore.

        ``asyncio.gather`` preserves input order, so the returned list aligns
        1-to-1 with ``docs``.
        """
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        tasks = [
            self._process_doc_loglikelihood_async(
                doc=doc,
                context_str=self.prompt_manager._prepare_plain_text(doc),
                semaphore=semaphore,
            )
            for doc in docs
        ]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Provider compatibility guard
    # ------------------------------------------------------------------

    def _check_text_completion_support(self) -> None:
        """Warn if the model is known to be chat-only and cannot serve loglikelihoods.

        ``loglikelihood`` and ``loglikelihood_rolling`` use
        ``litellm.atext_completion`` (``/v1/completions``) with ``echo=True``
        and ``logprobs=1``.  Models whose ``mode`` is ``"chat"`` in litellm's
        registry (gpt-4o, Claude, Gemini, …) do not expose this endpoint and
        will return all-``-inf`` results.

        We use ``litellm.get_model_info()`` so we only warn for *positively
        identified* chat-only models, avoiding false positives on completion
        models like ``gpt-3.5-turbo-instruct`` whose params list may not
        enumerate ``echo`` explicitly.
        """
        try:
            model_info = litellm.get_model_info(model=self.model, custom_llm_provider=self.provider) or {}
            mode = model_info.get("mode", "")
            if mode == "chat":
                logger.warning(
                    f"Model '{self.model}' is registered as a chat-only model (mode='chat'). "
                    "loglikelihood and loglikelihood_rolling require the /v1/completions "
                    "endpoint with echo=True — chat-only models do not support this. "
                    "Results will be all -inf. "
                    "Use 'gpt-3.5-turbo-instruct' or any OpenAI-compatible local server "
                    "(llama.cpp, vLLM serve, etc.) instead."
                )
        except Exception:
            pass  # Registry lookup failed — proceed silently, never crash an eval

    def _warn_if_too_long(self, text: str, label: str = "") -> None:
        """Warn when ``text`` is estimated to exceed the model's context window.

        Uses tiktoken (via ``litellm.encode``) for a fast, local token count.
        Silently skips if ``max_length`` is unknown or encoding fails.
        """
        try:
            n_tokens = len(encode(self.model, text))
            limit = self.max_length
            if limit and n_tokens > limit:
                tag = f" [{label}]" if label else ""
                logger.warning(
                    f"Input{tag} is ~{n_tokens} tokens, which exceeds max_length={limit}. "
                    "The API may truncate or reject the request. "
                    "Consider shortening your context or choice strings."
                )
        except Exception:
            pass  # Encoding unavailable for this model — skip silently

    # ------------------------------------------------------------------
    # loglikelihood (MCQ / log-prob ranking)
    # ------------------------------------------------------------------

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Compute log-likelihoods for MCQ-style tasks via ``litellm.text_completion``.

        Uses ``echo=True`` and ``logprobs=1`` to retrieve per-token log
        probabilities for the full (context + choice) string, then isolates the
        continuation slice with the Token Alignment Engine (see
        ``_find_continuation_start``).

        Provider requirement:
            The underlying model must support the ``/v1/completions`` endpoint
            with ``echo`` and ``logprobs`` parameters — for example
            ``gpt-3.5-turbo-instruct`` or any OpenAI-compatible local server
            (llama.cpp, vLLM serving, etc.).  Chat-only models such as
            ``gpt-4o`` or Claude do not expose this endpoint and are **not**
            supported by this method.
        """
        self._check_text_completion_support()

        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Loglikelihood splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            split_docs = list(split)
            split_results = asyncio.run(self._loglikelihood_async(split_docs))
            results.extend(split_results)

        return dataset.get_original_order(results)

    # ------------------------------------------------------------------
    # loglikelihood_rolling (perplexity)
    # ------------------------------------------------------------------

    async def _process_doc_rolling_async(
        self,
        doc: Doc,
        semaphore: asyncio.Semaphore,
    ) -> ModelResponse:
        """Compute per-token log-probabilities for the entire document text.

        Sends the full document as the prompt with ``echo=True`` and collects
        one logprob per token (skipping the leading null and the trailing
        generated token appended by ``max_tokens=1``).

        The returned ``ModelResponse.logprobs`` is a list of per-token floats.
        ``PerplexityPreparator`` sums this list to obtain the document-level
        log-likelihood, which is then used to compute perplexity, weighted
        perplexity, or bits-per-byte.
        """
        doc_text = self.prompt_manager._prepare_plain_text(doc)
        self._warn_if_too_long(doc_text, label=f"doc '{doc.id}'")
        response = await self._call_api_text_completion_async(doc_text, semaphore)

        if response is None or not getattr(response, "choices", None):
            return ModelResponse(input=doc_text, logprobs=[float("-inf")])

        lp_obj = getattr(response.choices[0], "logprobs", None)
        if lp_obj is None or not getattr(lp_obj, "token_logprobs", None):
            return ModelResponse(input=doc_text, logprobs=[float("-inf")])

        token_logprobs: list = list(lp_obj.token_logprobs or [])

        # token_logprobs[0]  → always None  (first token has no prior context)
        # token_logprobs[1:-1] → per-token log-probs for the full document
        # token_logprobs[-1] → the 1 newly generated token from max_tokens=1 (discard)
        rolling_logprobs = [v for v in token_logprobs[1:-1] if v is not None]

        return ModelResponse(
            input=doc_text,
            logprobs=rolling_logprobs,
        )

    async def _loglikelihood_rolling_async(self, docs: list[Doc]) -> list[ModelResponse]:
        """Async coordinator for rolling perplexity: one API call per doc."""
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        tasks = [self._process_doc_rolling_async(doc=doc, semaphore=semaphore) for doc in docs]
        return list(await asyncio.gather(*tasks))

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """Compute rolling log-likelihoods for perplexity-style evaluation.

        Each document is sent as a single prompt with ``echo=True`` so that the
        API returns per-token log-probabilities for the whole text.  The result
        is a ``ModelResponse`` whose ``logprobs`` list holds one float per
        token; downstream preparators (``PerplexityPreparator``,
        ``TargetPerplexityPreparator``) sum these to produce the document-level
        log-likelihood used by perplexity / bits-per-byte metrics.

        Provider requirement: same as ``loglikelihood`` — requires the
        ``/v1/completions`` endpoint with ``echo`` and ``logprobs``.
        """
        self._check_text_completion_support()

        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Loglikelihood rolling splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            split_docs = list(split)
            split_results = asyncio.run(self._loglikelihood_rolling_async(split_docs))
            results.extend(split_results)

        return dataset.get_original_order(results)

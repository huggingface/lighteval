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
from dataclasses import replace
from typing import Coroutine, Optional

import requests
from huggingface_hub import TextGenerationInputGenerateParameters, TextGenerationInputGrammarType, TextGenerationOutput
from transformers.models.auto.tokenization_auto import AutoTokenizer

from lighteval.models.abstract_model import ModelConfig
from lighteval.models.endpoints.endpoint_model import InferenceEndpointModel
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.utils.cache_management import SampleCache
from lighteval.utils.imports import Extra, is_package_available, requires


if is_package_available(Extra.TGI):
    from text_generation import AsyncClient
else:
    from unittest.mock import Mock

    AsyncClient = Mock()


BATCH_SIZE = 50


def divide_chunks(array, n):
    # looping till length array
    for i in range(0, len(array), n):
        yield array[i : i + n]


class TGIModelConfig(ModelConfig):
    """Configuration class for Text Generation Inference (TGI) backend.

    doc: https://huggingface.co/docs/text-generation-inference/en/index

    This configuration is used to connect to TGI servers that serve HuggingFace models
    using the text-generation-inference library. TGI provides high-performance inference
    with features like continuous batching and efficient memory management.

    Attributes:
        inference_server_address (str | None):
            Address of the TGI server. Format: "http://host:port" or "https://host:port".
            Example: "http://localhost:8080"
        inference_server_auth (str | None):
            Authentication token for the TGI server. If None, no authentication is used.
        model_name (str | None):
            Optional model name override. If None, uses the model name from server info.
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.

    Example:
        ```python
        config = TGIModelConfig(
            inference_server_address="http://localhost:8080",
            inference_server_auth="your-auth-token",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    inference_server_address: str | None = None
    inference_server_auth: str | None = None
    model_name: str | None
    model_info: dict | None = None
    batch_size: int = 1


# inherit from InferenceEndpointModel instead of LightevalModel since they both use the same interface, and only overwrite
# the client functions, since they use a different client.
class ModelClient(InferenceEndpointModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: TGIModelConfig) -> None:
        headers = (
            {} if config.inference_server_auth is None else {"Authorization": f"Bearer {config.inference_server_auth}"}
        )

        self.client = AsyncClient(config.inference_server_address, headers=headers, timeout=240)
        self.generation_parameters = config.generation_parameters
        self.generation_config = TextGenerationInputGenerateParameters(**self.generation_parameters.to_tgi_ie_dict())
        self._max_gen_toks = 256
        self.model_info = requests.get(f"{config.inference_server_address}/info", headers=headers).json()
        if "model_id" not in self.model_info:
            raise ValueError("Error occurred when fetching info: " + str(self.model_info))
        if config.model_name:
            self.model_info["model_id"] = config.model_name
        else:
            # Set the model_name in config to the actual model_id from server for caching
            config.model_name = self.model_info["model_id"]
        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_id"])
        self._add_special_tokens = True
        self.use_async = True
        self.config.model_info = self.model_info

        # Initialize prompt manager (required by parent class)
        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    @requires(Extra.TGI)
    def _async_process_request(
        self,
        context: str,
        stop_tokens: list[str],
        max_tokens: int,
        grammar: Optional[TextGenerationInputGrammarType] = None,
    ) -> Coroutine[None, list[TextGenerationOutput], str]:
        # Todo: add an option to launch with conversational instead for chat prompts
        # We create a copy of the current text generation params
        generation_config: TextGenerationInputGenerateParameters = replace(
            self.generation_config,
            stop=stop_tokens,
            max_new_tokens=max_tokens,
            details=True,
            decoder_input_details=True,
            grammar=grammar,
        )

        generated_text = self.client.generate(
            prompt=context,
            do_sample=generation_config.do_sample or False,
            max_new_tokens=generation_config.max_new_tokens,
            best_of=generation_config.best_of,
            repetition_penalty=generation_config.repetition_penalty,
            return_full_text=generation_config.return_full_text or False,
            seed=generation_config.seed,
            stop_sequences=generation_config.stop,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            truncate=generation_config.truncate,
            typical_p=generation_config.typical_p,
            watermark=generation_config.watermark or False,
            decoder_input_details=generation_config.decoder_input_details,
            grammar=generation_config.grammar,
        )

        return generated_text

    @requires(Extra.TGI)
    def _process_request(self, *args, **kwargs) -> TextGenerationOutput:
        return asyncio.run(self._async_process_request(*args, **kwargs))

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
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return ModelClient._DEFAULT_MAX_LENGTH

    @property
    def disable_tqdm(self) -> bool:
        return False

    def cleanup(self):
        pass

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

# inherit from InferenceEndpointModel instead of LightevalModel since they both use the same interface, and only overwrite
# the client functions, since they use a different client.
import asyncio
from typing import Coroutine
from lighteval.models.endpoint_model import InferenceEndpointModel
from huggingface_hub import TextGenerationOutput
from lighteval.models.utils import retry_with_backoff
from transformers import AutoTokenizer
from openai import AsyncOpenAI

class OAIModelClient(InferenceEndpointModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, address, model_id, auth_token=None) -> None:
        self.client = AsyncOpenAI(base_url=address, api_key=(auth_token or "none"))
        self.model_id = model_id
        self._max_gen_toks = 256

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._add_special_tokens = True
        self.use_async = True

    async def _async_process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> Coroutine[None, TextGenerationOutput, str]:
        # Todo: add an option to launch with conversational instead for chat prompts
        output = await retry_with_backoff(self.client.completions.create(
            model="/repository", 
            prompt=context,
            max_tokens=max_tokens,
            stop=stop_tokens
        ))

        return TextGenerationOutput(generated_text=output.choices[0].text)
        
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
        return OAIModelClient._DEFAULT_MAX_LENGTH

    @property
    def disable_tqdm(self) -> bool:
        False

    def cleanup(self):
        pass

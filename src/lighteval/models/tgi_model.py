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

import requests
from huggingface_hub import AsyncInferenceClient, InferenceClient
from transformers import AutoTokenizer

from lighteval.models.endpoint_model import InferenceEndpointModel


BATCH_SIZE = 50


def divide_chunks(array, n):
    # looping till length array
    for i in range(0, len(array), n):
        yield array[i : i + n]


class ModelClient(InferenceEndpointModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, address, auth_token=None, model_id=None) -> None:
        headers = {} if auth_token is None else {"Authorization": f"Bearer {auth_token}"}

        self.client = InferenceClient(base_url=address, headers=headers, timeout=240)
        self.async_client = AsyncInferenceClient(base_url=address, headers=headers, timeout=240)
        self._max_gen_toks = 256
        self.model_info = requests.get(f"{address}/info", headers=headers).json()
        if "model_id" not in self.model_info:
            raise ValueError("Error occured when fetching info: " + str(self.model_info))
        if model_id:
            self.model_info["model_id"] = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_id"])
        self._add_special_tokens = True
        self.use_async = True
        self.name = self.model_info["model_id"]

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
        False

    def cleanup(self):
        pass

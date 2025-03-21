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

from dataclasses import asdict

import pytest

from lighteval.models.endpoints.tgi_model import TGIModelConfig


class TestTGIModelConfig:
    @pytest.mark.parametrize(
        "config_path, expected_config",
        [
            (
                "examples/model_configs/tgi_model.yaml",
                {
                    "inference_server_address": "",
                    "inference_server_auth": None,
                    "model_id": None,
                    "generation_parameters": {
                        "early_stopping": None,
                        "frequency_penalty": None,
                        "length_penalty": None,
                        "max_new_tokens": None,
                        "min_new_tokens": None,
                        "min_p": None,
                        "presence_penalty": None,
                        "repetition_penalty": None,
                        "seed": None,
                        "stop_tokens": None,
                        "temperature": None,
                        "top_k": None,
                        "top_p": None,
                        "truncate_prompt": None,
                        "response_format": None,
                    },
                },
            ),
        ],
    )
    def test_from_path(self, config_path, expected_config):
        config = TGIModelConfig.from_path(config_path)
        assert asdict(config) == expected_config

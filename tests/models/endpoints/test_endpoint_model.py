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

import pytest
import yaml

from lighteval.models.endpoints.endpoint_model import InferenceEndpointModelConfig


class TestInferenceEndpointModelConfig:
    @pytest.mark.parametrize(
        "config_path, expected_config",
        [
            (
                "examples/model_configs/endpoint_model.yaml",
                {
                    "model_name": "meta-llama/Llama-2-7b-hf",
                    "dtype": "float16",
                    "revision": "main",
                    "endpoint_name": None,
                    "reuse_existing": False,
                    "accelerator": "gpu",
                    "region": "eu-west-1",
                    "vendor": "aws",
                    "instance_type": "nvidia-a10g",
                    "instance_size": "x1",
                    "framework": "pytorch",
                    "endpoint_type": "protected",
                    "namespace": None,
                    "image_url": None,
                    "env_vars": None,
                    "add_special_tokens": True,
                    "generation_parameters": {
                        "early_stopping": None,
                        "frequency_penalty": None,
                        "length_penalty": None,
                        "max_new_tokens": 256,
                        "min_new_tokens": None,
                        "min_p": None,
                        "presence_penalty": None,
                        "repetition_penalty": None,
                        "seed": None,
                        "stop_tokens": None,
                        "temperature": 0.2,
                        "top_k": None,
                        "top_p": 0.9,
                        "truncate_prompt": None,
                        "response_format": None,
                    },
                },
            ),
        ],
    )
    def test_from_path(self, config_path, expected_config):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config = InferenceEndpointModelConfig.from_path(config_path)
        assert config.model_dump() == expected_config

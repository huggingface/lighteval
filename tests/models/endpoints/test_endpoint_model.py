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

from lighteval.models.endpoints.endpoint_model import InferenceEndpointModelConfig


# "examples/model_configs/endpoint_model.yaml"


class TestInferenceEndpointModelConfig:
    @pytest.mark.parametrize(
        "config_path, expected_config",
        [
            (
                "examples/model_configs/endpoint_model.yaml",
                {
                    "model_name": "meta-llama/Llama-2-7b-hf",
                    "revision": "main",
                    "model_dtype": "float16",
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
                },
            ),
            (
                "examples/model_configs/serverless_model.yaml",
                {
                    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                    # Defaults:
                    "revision": "main",
                    "model_dtype": None,
                    "endpoint_name": None,
                    "reuse_existing": False,
                    "accelerator": "gpu",
                    "region": "us-east-1",
                    "vendor": "aws",
                    "instance_type": None,
                    "instance_size": None,
                    "framework": "pytorch",
                    "endpoint_type": "protected",
                    "namespace": None,
                    "image_url": None,
                    "env_vars": None,
                },
            ),
            (
                "examples/model_configs/endpoint_model_reuse_existing.yaml",
                {"endpoint_name": "llama-2-7B-lighteval", "reuse_existing": True},
            ),
        ],
    )
    def test_from_path(self, config_path, expected_config):
        config = InferenceEndpointModelConfig.from_path(config_path)
        for key, value in expected_config.items():
            assert getattr(config, key) == value

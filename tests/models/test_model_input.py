# MIT License

# Copyright (c) 2025 The HuggingFace Team

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

from lighteval.models.model_input import GenerationParameters


class TestGenerationParameters:
    @pytest.mark.parametrize(
        "model_args, expected",
        [
            (
                "generation_parameters={temperature: 0.7,top_p: 0.95},pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,max_model_length=32768,gpu_memory_utilisation=0.8",
                {"temperature": 0.7, "top_p": 0.95},
            ),
            (
                "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,generation_parameters={temperature: 0.7,top_p: 0.95},max_model_length=32768,gpu_memory_utilisation=0.8",
                {"temperature": 0.7, "top_p": 0.95},
            ),
            (
                "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,max_model_length=32768,gpu_memory_utilisation=0.8,generation_parameters={temperature: 0.7,top_p: 0.95}",
                {"temperature": 0.7, "top_p": 0.95},
            ),
        ],
    )
    def test_extract_num_samples(self, model_args: str, expected):
        gen = GenerationParameters.from_model_args(model_args)
        for k, v in expected.items():
            assert getattr(gen, k) == v

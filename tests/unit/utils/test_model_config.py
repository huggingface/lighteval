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

import unittest

from lighteval.models.model_input import ChatTemplateParameters, GenerationParameters
from lighteval.models.utils import ModelConfig


class TestModelConfig(unittest.TestCase):
    def test_model_config_init(self):
        config = ModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            generation_parameters=GenerationParameters(temperature=0.7),
            system_prompt="You are a helpful assistant.",
            chat_template_parameters=ChatTemplateParameters(reasoning_effort="low"),
        )

        self.assertEqual(config.model_name, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(config.generation_parameters.temperature, 0.7)
        self.assertEqual(config.system_prompt, "You are a helpful assistant.")
        self.assertEqual(config.chat_template_parameters.reasoning_effort, "low")

    def test_model_config_init_command_line(self):
        config = ModelConfig.from_args(
            'model_name=meta-llama/Llama-3.1-8B-Instruct,system_prompt="You are a helpful assistant.",generation_parameters={temperature:0.7},chat_template_parameters={reasoning_effort:low}'
        )

        self.assertEqual(config.model_name, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(config.generation_parameters.temperature, 0.7)
        self.assertEqual(config.system_prompt, '"You are a helpful assistant."')  # is this what we want?
        self.assertEqual(config.chat_template_parameters.reasoning_effort, "low")

    def test_model_config_generation_parameters_parse_single_int(self):
        config = ModelConfig.from_args(
            "model_name=meta-llama/Llama-3.1-8B-Instruct,generation_parameters={temperature:0.7}"
        )
        self.assertEqual(config.generation_parameters.temperature, 0.7)

    def test_model_config_generation_parameters_parse_multiple_int(self):
        config = ModelConfig.from_args(
            "model_name=meta-llama/Llama-3.1-8B-Instruct,generation_parameters={temperature:0.7,top_k:42}"
        )
        self.assertEqual(config.generation_parameters.temperature, 0.7)
        self.assertEqual(config.generation_parameters.top_k, 42)

    @unittest.skip("This is not working at this time")
    def test_model_config_generation_parameters_parse_string(self):
        config = ModelConfig.from_args(
            'model_name=meta-llama/Llama-3.1-8B-Instruct,generation_parameters={response_format:{"type":"json_object"}}'
        )
        self.assertEqual(config.generation_parameters.temperature, 0.7)

    @unittest.skip("This is not working at this time")
    def test_model_config_chat_template_parameters_parse_single_int(self):
        config = ModelConfig.from_args(
            "model_name=meta-llama/Llama-3.1-8B-Instruct,chat_template_parameters={temperature:0.7}"
        )
        self.assertEqual(config.chat_template_parameters.temperature, 0.7)

    def test_model_config_chat_template_parameters_parse_string(self):
        config = ModelConfig.from_args(
            "model_name=meta-llama/Llama-3.1-8B-Instruct,chat_template_parameters={reasoning_effort:low}"
        )
        self.assertEqual(config.chat_template_parameters.reasoning_effort, "low")

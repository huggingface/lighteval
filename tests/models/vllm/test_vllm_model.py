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

import unittest
from unittest.mock import Mock, patch

from transformers import AutoTokenizer

from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig


class TestVLLMTokenizerCreation(unittest.TestCase):
    def test_tokenizer_created_with_correct_revision(self):
        config = VLLMModelConfig(
            model_name="lighteval/different-chat-templates-per-revision", revision="new_chat_template"
        )
        vllm_tokenizer = VLLMModel.__new__(VLLMModel)._create_auto_tokenizer(config)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.revision,
        )
        self.assertEqual(vllm_tokenizer.chat_template, tokenizer.chat_template)


class TestVLLMModelUseChatTemplate(unittest.TestCase):
    @patch("lighteval.models.vllm.vllm_model.VLLMModel._create_auto_model")
    def test_vllm_model_use_chat_template_with_different_model_names(self, mock_create_model):
        """Test that VLLMModel correctly calls uses_chat_template with different model names."""
        test_cases = [
            ("Qwen/Qwen3-0.6B", True),
            ("gpt2", False),
        ]

        for model_name, expected_result in test_cases:
            with self.subTest(model_name=model_name):
                # We skip the model creation phase
                mock_create_model.return_value = Mock()

                config = VLLMModelConfig(model_name=model_name)
                model = VLLMModel(config)

                self.assertEqual(model.use_chat_template, expected_result)
                self.assertEqual(model.use_chat_template, model._tokenizer.chat_template is not None)

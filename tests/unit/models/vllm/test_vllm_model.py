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

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from transformers import AutoTokenizer

from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig, build_vllm_token_prompts
from lighteval.tasks.requests import Doc


class TestVLLMPromptConstruction(unittest.TestCase):
    def test_build_vllm_token_prompts_uses_tokens_prompt_when_available(self):
        fake_inputs = ModuleType("vllm.inputs")
        fake_inputs.TokensPrompt = lambda *, prompt_token_ids: {  # noqa: E731
            "kind": "tokens_prompt",
            "prompt_token_ids": prompt_token_ids,
        }
        fake_vllm = ModuleType("vllm")
        fake_vllm.inputs = fake_inputs

        with patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.inputs": fake_inputs}):
            prompts = build_vllm_token_prompts([[1, 2], [3]])

        self.assertEqual(
            prompts,
            [
                {"kind": "tokens_prompt", "prompt_token_ids": [1, 2]},
                {"kind": "tokens_prompt", "prompt_token_ids": [3]},
            ],
        )


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


class TestVLLMModelPromptTokens(unittest.TestCase):
    @patch("lighteval.models.vllm.vllm_model.VLLMModel._create_auto_model")
    def test_prompt_starts_with_one_bos(self, mock_create_model):
        """The token ids sent to vLLM start with exactly one BOS: chat templates that write BOS (Llama 3) do not get
        a second one from the tokenizer, plain-text prompts keep the tokenizer's BOS, and tokenizers that add none
        (Qwen) are unchanged."""

        def generate(prompts, sampling_params, use_tqdm):
            return [
                Mock(outputs=[Mock(token_ids=[], text="")], prompt_token_ids=p["prompt_token_ids"]) for p in prompts
            ]

        mock_create_model.return_value = Mock(generate=generate)
        doc = Doc(query="What is 2 + 2?", choices=[""], gold_index=0, generation_size=1, stop_sequences=[])
        test_cases = [
            ("unsloth/Llama-3.2-1B-Instruct", {}, [128000, 128006]),  # chat: <|begin_of_text|><|start_header_id|>
            ("unsloth/Llama-3.2-1B-Instruct", {"override_chat_template": False}, [128000, 3923]),  # plain text
            ("Qwen/Qwen3-0.6B", {}, [151644, 872]),  # chat, no BOS: <|im_start|>user
        ]

        for model_name, extra_args, expected_start in test_cases:
            with self.subTest(model_name=model_name, **extra_args):
                config = VLLMModelConfig(model_name=model_name, **extra_args)
                model = VLLMModel(config)

                (response,) = model._greedy_until([doc])
                self.assertEqual(response.input_tokens[:2], expected_start)

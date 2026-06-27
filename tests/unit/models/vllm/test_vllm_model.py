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

from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig, build_vllm_token_prompts


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


class _RecordingSamplingParams:
    """Stand-in for ``vllm.SamplingParams`` that records constructor kwargs as attributes.

    ``VLLMModel._generate`` builds a ``SamplingParams`` and then mutates a few fields on it, so a
    plain ``Mock`` would happily accept anything and hide whether the right values are set. This
    keeps the values around so they can be asserted on.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestVLLMModelConfigWiring(unittest.TestCase):
    """``VLLMModel.__init__`` should copy config values onto the attributes the rest of the class
    reads. Construction is kept offline by skipping model creation and the chat-template lookup, so
    no GPU, vllm install, or network is needed."""

    def _build_model(self, config: VLLMModelConfig, tokenizer=None) -> VLLMModel:
        with (
            patch.object(VLLMModel, "_create_auto_model", return_value=Mock()),
            patch.object(VLLMModel, "_create_auto_tokenizer", return_value=tokenizer or Mock(chat_template=None)),
            patch("lighteval.models.vllm.vllm_model.uses_chat_template", return_value=False),
        ):
            return VLLMModel(config)

    def test_add_special_tokens_is_taken_from_config(self):
        self.assertTrue(
            self._build_model(VLLMModelConfig(model_name="gpt2", add_special_tokens=True)).add_special_tokens
        )
        self.assertFalse(
            self._build_model(VLLMModelConfig(model_name="gpt2", add_special_tokens=False)).add_special_tokens
        )

    def test_max_length_is_taken_from_config(self):
        self.assertEqual(self._build_model(VLLMModelConfig(model_name="gpt2", max_model_length=4096)).max_length, 4096)

    def test_max_length_defaults_to_none_when_unset(self):
        # Left as None at construction; it is filled in later from the engine inside _create_auto_model.
        self.assertIsNone(self._build_model(VLLMModelConfig(model_name="gpt2")).max_length)

    def test_tokenizer_property_returns_the_created_tokenizer(self):
        sentinel = Mock(chat_template=None)
        self.assertIs(self._build_model(VLLMModelConfig(model_name="gpt2"), tokenizer=sentinel).tokenizer, sentinel)


class TestVLLMGenerateSamplingParams(unittest.TestCase):
    """``VLLMModel._generate`` is the boundary with vllm's ``SamplingParams`` API, so these tests
    pin down how the runtime arguments map onto it -- the kind of silent breakage #721 was about."""

    def _bare_model(self, config: VLLMModelConfig | None = None) -> VLLMModel:
        # __new__ avoids the heavy __init__; we only set the attributes _generate touches.
        model = VLLMModel.__new__(VLLMModel)
        model.data_parallel_size = 1
        model.model = Mock()
        if config is not None:
            model.config = config
        return model

    def test_scoring_pass_uses_fixed_greedy_params(self):
        # generate=False is the loglikelihood/scoring path: greedy, a single token, prompt logprobs kept.
        model = self._bare_model()
        with (
            patch("lighteval.models.vllm.vllm_model.SamplingParams", _RecordingSamplingParams),
            patch("lighteval.models.vllm.vllm_model.build_vllm_token_prompts", side_effect=lambda inputs: inputs),
        ):
            model._generate(inputs=[[1, 2, 3]], generate=False)

        call = model.model.generate.call_args
        sampling_params = call.kwargs["sampling_params"]
        self.assertEqual(sampling_params.temperature, 0.0)
        self.assertEqual(sampling_params.prompt_logprobs, 1)
        self.assertEqual(sampling_params.max_tokens, 1)
        self.assertFalse(sampling_params.detokenize)
        self.assertEqual(call.kwargs["prompts"], [[1, 2, 3]])

    def test_generation_pass_maps_runtime_args_onto_sampling_params(self):
        model = self._bare_model(
            VLLMModelConfig(model_name="gpt2", generation_parameters=GenerationParameters(temperature=0.7))
        )
        with (
            patch("lighteval.models.vllm.vllm_model.SamplingParams", _RecordingSamplingParams),
            patch("lighteval.models.vllm.vllm_model.build_vllm_token_prompts", side_effect=lambda inputs: inputs),
        ):
            model._generate(
                inputs=[[1, 2]], max_new_tokens=20, stop_tokens=["</s>"], returns_logits=True, num_samples=1
            )

        sampling_params = model.model.generate.call_args.kwargs["sampling_params"]
        self.assertEqual(sampling_params.n, 1)
        self.assertEqual(sampling_params.max_tokens, 20)
        self.assertEqual(sampling_params.stop, ["</s>"])
        self.assertEqual(sampling_params.logprobs, 1)  # returns_logits=True -> 1, otherwise 0
        self.assertEqual(sampling_params.temperature, 0.7)  # carried over from generation_parameters

    def test_sampling_multiple_completions_at_zero_temperature_raises(self):
        # num_samples > 1 with greedy decoding (temperature=0) is ill-defined; reject it early.
        model = self._bare_model(
            VLLMModelConfig(model_name="gpt2", generation_parameters=GenerationParameters(temperature=0.0))
        )
        with patch("lighteval.models.vllm.vllm_model.SamplingParams", _RecordingSamplingParams):
            with self.assertRaises(ValueError):
                model._generate(inputs=[[1, 2]], num_samples=2)

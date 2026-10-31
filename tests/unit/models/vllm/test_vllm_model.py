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

from lighteval.models.vllm import vllm_model
from lighteval.models.vllm.vllm_model import (
    VLLMModel,
    VLLMModelConfig,
    build_vllm_token_prompts,
    gather_with_progress_bar,
    run_inference_one_model,
)
from lighteval.utils.imports import is_package_available


class FakeTqdm:
    """Records how the main process progress bar is created and updated."""

    def __init__(self, *args, total=None, disable=False, **kwargs):
        self.total = total
        self.disable = disable
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, n):
        self.updates.append(n)


class FakeRay:
    """Minimal ray stand-in returning workers in the reverse of their submission order."""

    def __init__(self):
        self.wait_calls = 0

    def wait(self, object_refs, num_returns=1):
        self.wait_calls += 1
        return object_refs[-num_returns:], object_refs[:-num_returns]

    def get(self, object_refs):
        return [f"result_{ref}" for ref in object_refs]


class FakeRayCluster(FakeRay):
    """Ray stand-in running the remote tasks eagerly, object references being task indices."""

    def __init__(self):
        super().__init__()
        self.task_results = []

    def remote(self, **kwargs):
        def decorator(function):
            cluster = self

            class RemoteFunction:
                @staticmethod
                def remote(*args):
                    cluster.task_results.append(function(*args))
                    return len(cluster.task_results) - 1

            return RemoteFunction

        return decorator

    def get(self, object_refs):
        return [self.task_results[ref] for ref in object_refs]

    def shutdown(self):
        pass


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


class TestVLLMDataParallelProgressBar(unittest.TestCase):
    def test_worker_progress_bars_are_disabled(self):
        """Worker bars are duplicated across data parallel workers, only the main process logs progress."""
        llm = Mock()
        with (
            patch.object(vllm_model, "LLM", return_value=llm),
            patch.object(vllm_model, "build_vllm_token_prompts", side_effect=list),
        ):
            run_inference_one_model({"model": "gpt2"}, Mock(), [[1, 2], [3]])

        self.assertIs(llm.generate.call_args.kwargs["use_tqdm"], False)

    def _gather(self, object_refs, prompt_counts, disable=False):
        """Calls gather_with_progress_bar with ray and tqdm faked out."""
        bars = []
        fake_ray = FakeRay()

        def tqdm_factory(*args, **kwargs):
            bars.append(FakeTqdm(*args, **kwargs))
            return bars[-1]

        with patch.object(vllm_model, "ray", fake_ray), patch.object(vllm_model, "tqdm", tqdm_factory):
            results = gather_with_progress_bar(object_refs, prompt_counts, disable=disable)

        return results, bars, fake_ray

    def test_single_progress_bar_tracks_all_prompts(self):
        results, bars, fake_ray = self._gather(["ref_0", "ref_1", "ref_2"], [4, 3, 2])

        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].total, 9)
        self.assertFalse(bars[0].disable)
        # Workers complete in reverse order here, each advances the bar by its own number of prompts
        self.assertEqual(bars[0].updates, [2, 3, 4])
        self.assertEqual(fake_ray.wait_calls, 3)
        # Results stay in the order the workers were submitted in
        self.assertEqual(results, ["result_ref_0", "result_ref_1", "result_ref_2"])

    def test_progress_bar_can_be_disabled(self):
        _, bars, _ = self._gather(["ref_0"], [1], disable=True)

        self.assertTrue(bars[0].disable)

    @unittest.skipUnless(is_package_available("vllm"), "vllm is not installed")
    def test_data_parallel_generate_only_logs_on_main_process(self):
        model = VLLMModel.__new__(VLLMModel)
        model.config = VLLMModelConfig(model_name="gpt2", data_parallel_size=2)
        model.data_parallel_size = 2
        model.tensor_parallel_size = 1
        model.model_args = {"model": "gpt2"}

        llm = Mock()
        llm.generate.side_effect = lambda prompts, sampling_params, use_tqdm: list(prompts)
        bars = []

        def tqdm_factory(*args, **kwargs):
            bars.append(FakeTqdm(*args, **kwargs))
            return bars[-1]

        with (
            patch.object(vllm_model, "ray", FakeRayCluster()),
            patch.object(vllm_model, "tqdm", tqdm_factory),
            patch.object(vllm_model, "LLM", return_value=llm),
            patch.object(vllm_model, "build_vllm_token_prompts", side_effect=list),
        ):
            outputs = model._generate(inputs=[[1], [2], [3], [4]], max_new_tokens=1, stop_tokens=[])

        # One bar on the main process for all prompts, none on the workers
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].total, 4)
        self.assertEqual(sum(bars[0].updates), 4)
        self.assertEqual(llm.generate.call_count, 2)
        for call in llm.generate.call_args_list:
            self.assertIs(call.kwargs["use_tqdm"], False)
        # Interleaved requests are still returned in the original order
        self.assertEqual(outputs, [[1], [2], [3], [4]])


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

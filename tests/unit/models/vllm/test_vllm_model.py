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

import importlib
import sys
import types
import unittest
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch

import pytest
from transformers import AutoTokenizer

from lighteval.models.model_input import GenerationParameters
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


# ---------------------------------------------------------------------------
# Unit tests for the pure-Python logic of ``VLLMModel`` (issue #724).
#
# ``vllm`` is a heavy GPU dependency that is not available on CPU CI runners, and
# ``VLLMModel`` is gated behind ``@requires("vllm")`` (it becomes a placeholder
# that raises on instantiation when vllm is missing). To exercise the model's own
# logic - the boundary with vllm's ``SamplingParams`` and the loglikelihood
# math - without a GPU, the ``vllm_model_module`` fixture below injects a light
# fake ``vllm`` package and reloads the module so the real class is available and
# bound to a recording ``SamplingParams`` stand-in. Instances are built with
# ``__new__`` so that ``__init__`` (which would spin up a real engine) is skipped
# and only the attributes each method needs are set.
# ---------------------------------------------------------------------------


class FakeSamplingParams:
    """Records the parameters ``VLLMModel`` passes to vllm's ``SamplingParams``.

    Only the attributes touched by ``VLLMModel._generate`` are modelled. The
    constructor stores everything it is given, and the model is free to set the
    remaining fields afterwards, exactly as it does with the real class.
    """

    def __init__(self, **kwargs):
        self.n = 1
        self.temperature = 1.0
        self.max_tokens = None
        self.stop = None
        self.logprobs = None
        self.prompt_logprobs = None
        self.detokenize = True
        self.__dict__.update(kwargs)
        self.init_kwargs = dict(kwargs)


def _fake_vllm_modules():
    """Build a minimal fake ``vllm`` (and ``ray``) package tree for imports."""
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = MagicMock(name="LLM")
    fake_vllm.RequestOutput = MagicMock(name="RequestOutput")
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_inputs = types.ModuleType("vllm.inputs")
    fake_inputs.TokensPrompt = lambda *, prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    fake_vllm.inputs = fake_inputs

    fake_parallel = types.ModuleType("vllm.distributed.parallel_state")
    fake_parallel.destroy_distributed_environment = MagicMock()
    fake_parallel.destroy_model_parallel = MagicMock()
    fake_distributed = types.ModuleType("vllm.distributed")
    fake_distributed.parallel_state = fake_parallel

    fake_async = types.ModuleType("vllm.v1.engine.async_llm")
    fake_async.AsyncLLM = MagicMock()
    fake_async.AsyncEngineArgs = MagicMock()

    fake_tokenizers = types.ModuleType("vllm.tokenizers")
    fake_tokenizers.get_tokenizer = MagicMock()

    fake_ray = types.ModuleType("ray")
    fake_ray.remote = lambda *args, **kwargs: lambda fn: fn
    fake_ray.get = MagicMock()
    fake_ray.shutdown = MagicMock()

    modules = {
        "vllm": fake_vllm,
        "vllm.inputs": fake_inputs,
        "vllm.distributed": fake_distributed,
        "vllm.distributed.parallel_state": fake_parallel,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": fake_async,
        "vllm.tokenizers": fake_tokenizers,
        "ray": fake_ray,
    }
    try:
        import more_itertools  # noqa: F401
    except ImportError:
        fake_more_itertools = types.ModuleType("more_itertools")
        fake_more_itertools.distribute = lambda n, iterable: [list(iterable)]
        modules["more_itertools"] = fake_more_itertools
    return modules


@pytest.fixture
def vllm_model_module():
    """Reload ``vllm_model`` with a fake ``vllm`` backend so the real class loads.

    This keeps the tests runnable on CPU while still exercising ``VLLMModel``'s
    own code. When vllm *is* installed (GPU CI) the fakes simply shadow it for the
    duration of the test; the module is reloaded again on teardown to restore the
    original import state for the rest of the suite.

    Yields:
        module: The reloaded ``vllm_model`` module exposing the real ``VLLMModel``.
    """
    import lighteval.models.vllm.vllm_model as vllm_model
    import lighteval.utils.imports as imports_mod

    real_is_available = imports_mod.is_package_available

    def fake_is_available(package):
        # ``package`` may be the raw string or a parsed ``Requirement`` object.
        name = getattr(package, "name", package)
        if name == "vllm":
            return True
        return real_is_available(package)

    with patch.dict(sys.modules, _fake_vllm_modules()):
        with patch.object(imports_mod, "is_package_available", fake_is_available):
            module = importlib.reload(vllm_model)
            try:
                yield module
            finally:
                pass
    # Restore the original (un-faked) module for any other tests importing it.
    importlib.reload(vllm_model)


def _new_model(module):
    """Create a ``VLLMModel`` without running ``__init__``."""
    return module.VLLMModel.__new__(module.VLLMModel)


class FakeLogprob:
    def __init__(self, logprob, rank):
        self.logprob = logprob
        self.rank = rank


class FakeGenerateOutput:
    def __init__(self, prompt_token_ids, prompt_logprobs):
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs


class TestVLLMProperties:
    def test_max_length_returns_configured_value(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model._max_length = 2048
        assert model.max_length == 2048

    def test_add_special_tokens_returns_configured_value(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model._add_special_tokens = True
        assert model.add_special_tokens is True

    def test_tokenizer_property_returns_backing_tokenizer(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        sentinel = object()
        model._tokenizer = sentinel
        assert model.tokenizer is sentinel


class TestVLLMGenerateSamplingParams:
    """The vllm boundary that PR #721 showed can break silently."""

    def test_scoring_path_uses_greedy_sampling_params(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model.data_parallel_size = 1
        model.model = MagicMock()
        model.model.generate.return_value = ["sentinel"]

        outputs = model._generate(inputs=[[1, 2, 3]], generate=False)

        assert outputs == ["sentinel"]
        call = model.model.generate.call_args
        sampling_params = call.kwargs["sampling_params"]
        assert sampling_params.temperature == 0.0
        assert sampling_params.prompt_logprobs == 1
        assert sampling_params.max_tokens == 1
        assert sampling_params.detokenize is False
        # Token ids are wrapped through build_vllm_token_prompts.
        assert call.kwargs["prompts"] == [{"prompt_token_ids": [1, 2, 3]}]
        assert call.kwargs["use_tqdm"] is True

    def test_generation_path_maps_sampling_fields(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model.config = vllm_model_module.VLLMModelConfig(model_name="gpt2")
        model.data_parallel_size = 1
        model.model = MagicMock()
        model.model.generate.return_value = []

        model._generate(
            inputs=[[1, 2]],
            max_new_tokens=32,
            stop_tokens=["</s>"],
            returns_logits=True,
            num_samples=1,
            generate=True,
        )

        sampling_params = model.model.generate.call_args.kwargs["sampling_params"]
        assert sampling_params.n == 1
        assert sampling_params.max_tokens == 32
        assert sampling_params.stop == ["</s>"]
        assert sampling_params.logprobs == 1

    def test_generation_path_disables_logprobs_when_not_requested(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model.config = vllm_model_module.VLLMModelConfig(model_name="gpt2")
        model.data_parallel_size = 1
        model.model = MagicMock()
        model.model.generate.return_value = []

        model._generate(inputs=[[1, 2]], returns_logits=False, num_samples=1, generate=True)

        sampling_params = model.model.generate.call_args.kwargs["sampling_params"]
        assert sampling_params.logprobs == 0

    def test_multiple_samples_with_greedy_temperature_raises(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        # Default generation parameters use temperature=0 (greedy).
        model.config = vllm_model_module.VLLMModelConfig(model_name="gpt2")
        model.data_parallel_size = 1
        model.model = MagicMock()

        with pytest.raises(ValueError):
            model._generate(inputs=[[1, 2]], num_samples=2, generate=True)

    def test_multiple_samples_allowed_with_positive_temperature(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model.config = vllm_model_module.VLLMModelConfig(
            model_name="gpt2",
            generation_parameters=GenerationParameters(temperature=0.7),
        )
        model.data_parallel_size = 1
        model.model = MagicMock()
        model.model.generate.return_value = []

        model._generate(inputs=[[1, 2]], num_samples=3, generate=True)

        sampling_params = model.model.generate.call_args.kwargs["sampling_params"]
        assert sampling_params.n == 3
        assert sampling_params.temperature == 0.7


class TestVLLMCreateAutoModel:
    def test_model_args_wiring_for_single_gpu(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model._max_length = 4096
        config = vllm_model_module.VLLMModelConfig(model_name="gpt2", max_model_length=4096, seed=99)

        returned = model._create_auto_model(config)

        assert returned is not None  # LLM(**model_args) was constructed
        args = model.model_args
        assert args["model"] == "gpt2"
        assert args["max_model_len"] == 4096
        assert args["seed"] == 99
        assert args["enforce_eager"] is True
        assert args["tensor_parallel_size"] == 1
        # Optional keys are only present when explicitly configured.
        assert "quantization" not in args
        assert "load_format" not in args

    def test_optional_quantization_and_load_format_are_forwarded(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model._max_length = 4096
        config = vllm_model_module.VLLMModelConfig(
            model_name="gpt2",
            max_model_length=4096,
            quantization="fp8",
            load_format="safetensors",
        )

        model._create_auto_model(config)

        assert model.model_args["quantization"] == "fp8"
        assert model.model_args["load_format"] == "safetensors"

    def test_subfolder_is_appended_to_revision(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model._max_length = 4096
        config = vllm_model_module.VLLMModelConfig(
            model_name="gpt2", max_model_length=4096, revision="abc", subfolder="sub"
        )

        model._create_auto_model(config)

        assert model.model_args["revision"] == "abc/sub"

    def test_data_parallel_defers_model_creation(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model._max_length = 4096
        config = vllm_model_module.VLLMModelConfig(model_name="gpt2", max_model_length=4096, data_parallel_size=2)

        returned = model._create_auto_model(config)

        # With data parallelism the model is built later by ray, so None is returned.
        assert returned is None
        assert model.model_args["distributed_executor_backend"] == "ray"
        assert model._batch_size == "auto"


class TestVLLMLoglikelihoodTokens:
    def test_continuation_logprobs_and_argmax(self, vllm_model_module):
        model = _new_model(vllm_model_module)
        model.pairwise_tokenization = False
        model.prompt_manager = MagicMock()
        model.prompt_manager.prepare_prompt.return_value = "CTX"
        # Context/continuation token ids for the two choices of the single doc.
        model.tok_encode_pair = MagicMock(return_value=([[10, 11], [10, 11]], [[20], [21, 22]]))

        # First choice: continuation [20] scored as top-1 (rank 1) -> logprob -0.5.
        # Second choice: continuation [21, 22]; 22 is not top-1 -> argmax False, sum -3.0.
        out0 = FakeGenerateOutput([10, 11, 20], [None, None, {20: FakeLogprob(-0.5, 1)}])
        out1 = FakeGenerateOutput(
            [10, 11, 21, 22],
            [None, None, {21: FakeLogprob(-1.0, 1)}, {22: FakeLogprob(-2.0, 3)}],
        )
        model._generate = MagicMock(return_value=[out0, out1])

        docs = [Doc(query="q", choices=["a", "bb"], gold_index=0)]
        responses = model._loglikelihood_tokens(docs)

        # Scoring must go through the greedy (generate=False) path.
        assert model._generate.call_args.kwargs["generate"] is False
        assert model._generate.call_args.args[0] == [[10, 11, 20], [10, 11, 21, 22]]

        assert len(responses) == 1
        response = responses[0]
        assert response.logprobs == [-0.5, -3.0]
        assert response.argmax_logits_eq_gold == [True, False]
        assert response.output_tokens == [[20], [21, 22]]
        assert response.input_tokens == [[10, 11], [10, 11]]
        assert response.input == "CTX"


class TestVLLMTokenPromptsBatching:
    def test_build_vllm_token_prompts_preserves_order_and_batch(self, vllm_model_module):
        prompts = vllm_model_module.build_vllm_token_prompts([[5, 6], [7], []])
        assert prompts == [
            {"prompt_token_ids": [5, 6]},
            {"prompt_token_ids": [7]},
            {"prompt_token_ids": []},
        ]

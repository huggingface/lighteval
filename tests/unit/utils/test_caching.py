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

import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch

import pytest
import torch

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache
from lighteval.utils.imports import Extra, is_package_available


class TestCaching(unittest.TestCase):
    def setUp(self):
        """Create simple test documents."""
        self.docs = []
        self.model_responses = []
        self.task_name = "cache_test"
        for i in range(3):
            doc = Doc(
                id=f"test_doc_{i}",
                task_name=self.task_name,
                query=f"Test question {i}: What is 2+2?",
                choices=["3", "4", "5", "6"],
                gold_index=1,
                instruction="Answer the math question",
            )
            model_resp = ModelResponse(
                input=doc.query,
                text=[f"Answer {i}"],
                input_tokens=[1, 2, 3, 4],
                output_tokens=[[5, 6, 7]],
                logprobs=[-0.1, -0.2, -0.3],
                argmax_logits_eq_gold=True,
            )
            self.docs.append(doc)
            self.model_responses.append(model_resp)

    def test_cache_directory_structure(self):
        """Test that cache directories are created correctly."""
        from lighteval.models.dummy.dummy_model import DummyModelConfig
        from lighteval.models.endpoints.endpoint_model import InferenceEndpointModelConfig
        from lighteval.models.endpoints.tgi_model import TGIModelConfig
        from lighteval.models.sglang.sglang_model import SGLangModelConfig
        from lighteval.models.transformers.transformers_model import TransformersModelConfig
        from lighteval.models.transformers.vlm_transformers_model import VLMTransformersModelConfig
        from lighteval.models.vllm.vllm_model import VLLMModelConfig

        # We skip AdapterModelConfig, DeltaModelConfig because of imports
        # We skip FullNanotronConfig as it's not standardized with our other configs, will need to be homogeneized
        model_configs = [
            TransformersModelConfig,
            VLMTransformersModelConfig,
            VLLMModelConfig,
            InferenceEndpointModelConfig,
            TGIModelConfig,
            SGLangModelConfig,
            DummyModelConfig,
        ]

        for model_config in model_configs:
            with self.subTest(model_config=model_config):
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_name = f"test_model_{model_config.__name__}"
                    # if model_config in [AdapterModelConfig, DeltaModelConfig]:
                    #    config = model_config(model_name=model_name, base_model=model_name + "2", cache_dir=temp_dir)
                    # else:
                    config = model_config(model_name=model_name, cache_dir=temp_dir)

                    # Create cache with custom directory
                    cache = SampleCache(config)

                    # Check directory structure
                    folder = cache.cache_dir
                    self.assertTrue(folder.exists())
                    self.assertIn(str(temp_dir), str(folder))
                    self.assertIn(model_name, str(folder))

    def test_cache_decorator_presence(self):
        """Test that @cached decorators are present on the right methods."""
        from lighteval.models.dummy.dummy_model import DummyModel
        from lighteval.models.endpoints.endpoint_model import InferenceEndpointModel
        from lighteval.models.endpoints.tgi_model import ModelClient
        from lighteval.models.nanotron.nanotron_model import NanotronLightevalModel
        from lighteval.models.sglang.sglang_model import SGLangModel
        from lighteval.models.transformers.adapter_model import AdapterModel
        from lighteval.models.transformers.delta_model import DeltaModel
        from lighteval.models.transformers.transformers_model import TransformersModel
        from lighteval.models.transformers.vlm_transformers_model import VLMTransformersModel
        from lighteval.models.vllm.vllm_model import AsyncVLLMModel, VLLMModel

        model_classes = [
            TransformersModel,
            AdapterModel,
            DeltaModel,
            VLMTransformersModel,
            VLLMModel,
            AsyncVLLMModel,
            InferenceEndpointModel,
            ModelClient,
            NanotronLightevalModel,
            SGLangModel,
            DummyModel,
        ]
        methods_to_check = ["greedy_until", "loglikelihood", "loglikelihood_rolling"]

        for model_class in model_classes:
            for method_name in methods_to_check:
                with self.subTest(model_class=model_class, method_name=method_name):
                    self.assertTrue(
                        hasattr(model_class, method_name), f"{method_name} method not found for {model_class}"
                    )
                    method = getattr(model_class, method_name)
                    # Check if method has been wrapped by @cached decorator
                    self.assertTrue(
                        hasattr(method, "__wrapped__"), f"{method_name} missing @cached decorator for {model_class}"
                    )

    def _test_cache(self, model: LightevalModel, test_cases):
        """Test that the @cached decorator logic works correctly - called by all model specific functions below."""
        for function_name, sampling_method in test_cases:
            with self.subTest(function_name=function_name):
                process_inputs = getattr(model, function_name)
                process_inputs(self.docs)

                cache: SampleCache = model._cache

                # Check task_id
                task_id = cache.get_task_id(self.task_name, sampling_method)
                self.assertEqual(task_id.task_name, self.task_name)
                self.assertEqual(task_id.sampling_method, sampling_method)

                # Verify cache files were created
                cache_file = cache.get_cache_path(task_id)
                self.assertTrue(cache_file.exists(), "Cache file not created")

                # Test retrieving from cache
                self.assertEqual(cache._load_cached_indices()[task_id], [doc.id for doc in self.docs])
                uncached_docs, tasks_with_cached_samples = cache.get_samples_to_process_and_cache(
                    docs=self.docs, sampling_method=sampling_method
                )
                self.assertEqual(tasks_with_cached_samples, {task_id})
                self.assertEqual(
                    len(uncached_docs), 0, f"{len(uncached_docs)} documents not found in cache when it should be 0"
                )

                # Verify cached results match original
                cached_responses = cache.get_samples_from_cache(
                    docs=self.docs, task_ids=[task_id], sampling_method=sampling_method
                )
                for cached_response, response in zip(cached_responses, self.model_responses):
                    self.assertEqual(asdict(cached_response), asdict(response))

    @patch("lighteval.models.transformers.transformers_model.TransformersModel._loglikelihood_tokens")
    @patch("lighteval.models.transformers.transformers_model.TransformersModel._padded_greedy_until")
    @patch("lighteval.models.transformers.transformers_model.Accelerator")
    @patch("lighteval.models.transformers.transformers_model.TransformersModel._create_auto_model")
    def test_cache_transformers(self, mock_create_model, mock_accelerator, mock_greedy_until, mock_loglikelihood):
        from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig

        # Skip the model creation phase
        mock_create_model = Mock()  # noqa F841

        # Mock accelerate related params
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.device = torch.device("cpu")
        mock_accelerator.return_value = mock_accelerator_instance

        mock_greedy_until.return_value = self.model_responses
        mock_loglikelihood.return_value = self.model_responses
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TransformersModelConfig(model_name="Qwen/Qwen3-0.6B", cache_dir=temp_dir)
            model = TransformersModel(config)

            self._test_cache(
                model,
                [
                    ("greedy_until", SamplingMethod.GENERATIVE),
                    ("loglikelihood", SamplingMethod.LOGPROBS),
                    ("loglikelihood_rolling", SamplingMethod.PERPLEXITY),
                ],
            )

    @patch("lighteval.models.vllm.vllm_model.VLLMModel._loglikelihood_tokens")
    @patch("lighteval.models.vllm.vllm_model.VLLMModel._greedy_until")
    @patch("lighteval.models.vllm.vllm_model.VLLMModel._create_auto_model")
    def test_cache_vllm(self, mock_create_model, mock_greedy_until, mock_loglikelihood):
        from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig

        # Mock VLLM LLM
        mock_create_model = Mock()  # noqa F841
        mock_greedy_until.return_value = self.model_responses
        mock_loglikelihood.return_value = self.model_responses

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VLLMModelConfig(model_name="Qwen/Qwen3-0.6B", cache_dir=temp_dir)
            model = VLLMModel(config)

            self._test_cache(
                model,
                [
                    ("greedy_until", SamplingMethod.GENERATIVE),
                    ("loglikelihood", SamplingMethod.LOGPROBS),
                ],
            )

    @patch("requests.get")
    @patch("lighteval.models.endpoints.tgi_model.ModelClient._greedy_until")
    @patch("lighteval.models.endpoints.tgi_model.ModelClient._loglikelihood")
    def test_cache_tgi(self, mock_loglikelihood, mock_greedy_until, mock_requests_get):
        from lighteval.models.endpoints.tgi_model import ModelClient, TGIModelConfig

        if not is_package_available(Extra.TGI):
            pytest.skip("Skipping because missing the imports")

        # Mock TGI requests
        mock_loglikelihood.return_value = self.model_responses
        mock_greedy_until.return_value = self.model_responses

        # Mock HTTP info request
        mock_requests_get.return_value.json.return_value = {"model_id": "Qwen/Qwen3-0.6B"}

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TGIModelConfig(
                model_name="Qwen/Qwen3-0.6B", cache_dir=temp_dir, inference_server_address="http://localhost:8080"
            )
            model = ModelClient(config)

            self._test_cache(
                model,
                [
                    ("greedy_until", SamplingMethod.GENERATIVE),
                    ("loglikelihood", SamplingMethod.LOGPROBS),
                    ("loglikelihood_rolling", SamplingMethod.PERPLEXITY),
                ],
            )

    @patch("lighteval.models.endpoints.endpoint_model.InferenceEndpointModel._loglikelihood")
    @patch("lighteval.models.endpoints.endpoint_model.InferenceEndpointModel._greedy_until")
    @patch("lighteval.models.endpoints.endpoint_model.InferenceEndpointModel._create_endpoint")
    def test_cache_endpoint(self, mock_init, mock_greedy_until, mock_loglikelihood):
        from lighteval.models.endpoints.endpoint_model import InferenceEndpointModel, InferenceEndpointModelConfig

        # Mock endpoint requests
        auto_model = Mock()
        auto_model.repository = "Qwen/Qwen3-0.6B"
        auto_model.revision = ""
        mock_init.return_value = auto_model, Mock(), Mock()  # noqa F841

        mock_greedy_until.return_value = self.model_responses
        mock_loglikelihood.return_value = self.model_responses

        with tempfile.TemporaryDirectory() as temp_dir:
            config = InferenceEndpointModelConfig(model_name="Qwen/Qwen3-0.6B", cache_dir=temp_dir)
            model = InferenceEndpointModel(config)

            self._test_cache(
                model,
                [
                    ("greedy_until", SamplingMethod.GENERATIVE),
                    ("loglikelihood", SamplingMethod.LOGPROBS),
                    ("loglikelihood_rolling", SamplingMethod.PERPLEXITY),
                ],
            )

    @patch("lighteval.models.sglang.sglang_model.SGLangModel._loglikelihood_tokens")
    @patch("lighteval.models.sglang.sglang_model.SGLangModel._greedy_until")
    @patch("lighteval.models.sglang.sglang_model.SGLangModel._create_auto_tokenizer")
    @patch("lighteval.models.sglang.sglang_model.SGLangModel._create_auto_model")
    def test_cache_sglang(
        self, mock_create_auto_model, mock_create_auto_tokenizer, mock_greedy_until, mock_loglikelihood
    ):
        from lighteval.models.sglang.sglang_model import SGLangModel, SGLangModelConfig

        # Mock SGLang engine
        mock_create_auto_model = Mock()  # noqa F841
        mock_create_auto_tokenizer = Mock()  # noqa F841
        mock_greedy_until.return_value = self.model_responses
        mock_loglikelihood.return_value = self.model_responses

        with tempfile.TemporaryDirectory() as temp_dir:
            config = SGLangModelConfig(model_name="Qwen/Qwen3-0.6B", cache_dir=temp_dir)
            model = SGLangModel(config)

            self._test_cache(
                model,
                [
                    ("greedy_until", SamplingMethod.GENERATIVE),
                    ("loglikelihood", SamplingMethod.LOGPROBS),
                ],
            )

    @patch("lighteval.models.transformers.vlm_transformers_model.VLMTransformersModel._greedy_until")
    @patch("lighteval.models.transformers.vlm_transformers_model.Accelerator")
    @patch("lighteval.models.transformers.vlm_transformers_model.VLMTransformersModel._create_auto_model")
    def test_cache_vlm_transformers(self, mock_create_model, mock_accelerator, mock_greedy_until):
        from lighteval.models.transformers.vlm_transformers_model import (
            VLMTransformersModel,
            VLMTransformersModelConfig,
        )

        # Mock accelerate related params
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.device = torch.device("cpu")
        mock_accelerator.return_value = mock_accelerator_instance

        # Skip the model creation phase
        mock_create_model = Mock()  # noqa F841
        mock_greedy_until.return_value = self.model_responses

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VLMTransformersModelConfig(model_name="HuggingFaceTB/SmolVLM-256M-Instruct", cache_dir=temp_dir)
            model = VLMTransformersModel(config)

            self._test_cache(
                model,
                [
                    ("greedy_until", SamplingMethod.GENERATIVE),
                ],
            )

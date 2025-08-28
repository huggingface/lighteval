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

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from datasets import Dataset
from huggingface_hub import HfApi

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.info_loggers import DetailsLogger

# ruff: noqa
from tests.fixtures import TESTING_EMPTY_HF_ORG_ID
from unittest.mock import patch, Mock
import unittest


@pytest.fixture
def mock_evaluation_tracker(request):
    passed_params = {}
    if request.keywords.get("evaluation_tracker"):
        passed_params = request.keywords["evaluation_tracker"].kwargs

    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs = {
            "output_dir": temp_dir,
            "save_details": passed_params.get("save_details", False),
            "push_to_hub": passed_params.get("push_to_hub", False),
            "push_to_tensorboard": passed_params.get("push_to_tensorboard", False),
            "hub_results_org": passed_params.get("hub_results_org", ""),
        }
        tracker = EvaluationTracker(**kwargs)
        tracker.general_config_logger.model_name = "test_model"

        # Create a dummy model config to prevent model_config.model_dump() errors
        from lighteval.models.dummy.dummy_model import DummyModelConfig

        dummy_model_config = DummyModelConfig(model_name="test_model")
        tracker.general_config_logger.log_model_info(model_config=dummy_model_config)

        yield tracker


@pytest.fixture
def mock_datetime(monkeypatch):
    mock_date = datetime(2023, 1, 1, 12, 0, 0)

    class MockDatetime:
        @classmethod
        def now(cls):
            return mock_date

        @classmethod
        def fromisoformat(cls, date_string: str):
            return mock_date

    monkeypatch.setattr("lighteval.logging.evaluation_tracker.datetime", MockDatetime)
    return mock_date


class TestLogging:
    def test_results_logging(self, mock_evaluation_tracker: EvaluationTracker):
        task_metrics = {
            "task1": {"accuracy": 0.8, "f1": 0.75},
            "task2": {"precision": 0.9, "recall": 0.85},
        }
        mock_evaluation_tracker.metrics_logger.metric_aggregated = task_metrics

        mock_evaluation_tracker.save()

        results_dir = Path(mock_evaluation_tracker.output_dir) / "results" / "test_model"
        assert results_dir.exists()

        result_files = list(results_dir.glob("results_*.json"))
        assert len(result_files) == 1

        with open(result_files[0], "r") as f:
            saved_results = json.load(f)

        assert "results" in saved_results
        assert saved_results["results"] == task_metrics
        assert saved_results["config_general"]["model_name"] == "test_model"

    def test_results_logging_template(self, mock_evaluation_tracker: EvaluationTracker):
        task_metrics = {
            "task1": {"accuracy": 0.8, "f1": 0.75},
            "task2": {"precision": 0.9, "recall": 0.85},
        }
        mock_evaluation_tracker.metrics_logger.metric_aggregated = task_metrics
        mock_evaluation_tracker.results_path_template = "{output_dir}/{org}_{model}"

        mock_evaluation_tracker.save()

        results_dir = Path(mock_evaluation_tracker.output_dir) / "_test_model"
        assert results_dir.exists()

        result_files = list(results_dir.glob("results_*.json"))
        assert len(result_files) == 1

        with open(result_files[0], "r") as f:
            saved_results = json.load(f)

        assert "results" in saved_results
        assert saved_results["results"] == task_metrics
        assert saved_results["config_general"]["model_name"] == "test_model"

    @pytest.mark.evaluation_tracker(save_details=True)
    def test_details_logging(self, mock_evaluation_tracker, mock_datetime):
        task_details = {
            "task1": [DetailsLogger.CompiledDetail(hashes=None, truncated=10, padded=5)],
            "task2": [DetailsLogger.CompiledDetail(hashes=None, truncated=20, padded=10)],
        }
        mock_evaluation_tracker.details_logger.details = task_details

        mock_evaluation_tracker.save()

        date_id = mock_datetime.isoformat().replace(":", "-")
        details_dir = Path(mock_evaluation_tracker.output_dir) / "details" / "test_model" / date_id
        assert details_dir.exists()

        for task in ["task1", "task2"]:
            file_path = details_dir / f"details_{task}_{date_id}.parquet"
            dataset = Dataset.from_parquet(str(file_path))
            assert len(dataset) == 1
            assert int(dataset[0]["truncated"]) == task_details[task][0].truncated
            assert int(dataset[0]["padded"]) == task_details[task][0].padded

    @pytest.mark.evaluation_tracker(save_details=False)
    def test_no_details_output(self, mock_evaluation_tracker: EvaluationTracker):
        mock_evaluation_tracker.save()

        details_dir = Path(mock_evaluation_tracker.output_dir) / "details" / "test_model"
        assert not details_dir.exists()

    @pytest.mark.skip(  # skipif
        reason="Secrets are not available in this environment",
        # condition=os.getenv("HF_TEST_TOKEN") is None,
    )
    @pytest.mark.evaluation_tracker(push_to_hub=True, hub_results_org=TESTING_EMPTY_HF_ORG_ID)
    def test_push_to_hub_works(
        self, testing_empty_hf_org_id, mock_evaluation_tracker: EvaluationTracker, mock_datetime
    ):
        # Prepare the dummy data
        task_metrics = {
            "task1": {"accuracy": 0.8, "f1": 0.75},
            "task2": {"precision": 0.9, "recall": 0.85},
        }
        mock_evaluation_tracker.metrics_logger.metric_aggregated = task_metrics

        task_details = {
            "task1": [DetailsLogger.CompiledDetail(truncated=10, padded=5)],
            "task2": [DetailsLogger.CompiledDetail(truncated=20, padded=10)],
        }
        mock_evaluation_tracker.details_logger.details = task_details
        mock_evaluation_tracker.save()

        # Verify using HfApi
        api = HfApi()

        # Check if repo exists and it's private
        expected_repo_id = f"{testing_empty_hf_org_id}/details_test_model_private"
        assert api.repo_exists(repo_id=expected_repo_id, repo_type="dataset")
        assert api.repo_info(repo_id=expected_repo_id, repo_type="dataset").private

        repo_files = api.list_repo_files(repo_id=expected_repo_id, repo_type="dataset")
        # Check if README.md exists
        assert any(file == "README.md" for file in repo_files)

        # Check that both results files were uploaded
        result_files = [file for file in repo_files if file.startswith("results_")]
        assert len(result_files) == 2
        assert len([file for file in result_files if file.endswith(".json")]) == 1
        assert len([file for file in result_files if file.endswith(".parquet")]) == 1

        # Check that the details dataset was uploaded
        details_files = [file for file in repo_files if "details_" in file and file.endswith(".parquet")]
        assert len(details_files) == 2


class TestProperties(unittest.TestCase):
    def setUp(self):
        # In setup in case we need to reuse for future tests
        from lighteval.models.dummy.dummy_model import DummyModelConfig
        from lighteval.models.endpoints.endpoint_model import (
            ServerlessEndpointModelConfig,
            InferenceEndpointModelConfig,
        )
        from lighteval.models.endpoints.inference_providers_model import InferenceProvidersModelConfig
        from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
        from lighteval.models.endpoints.tgi_model import TGIModelConfig
        from lighteval.models.sglang.sglang_model import SGLangModelConfig
        from lighteval.models.transformers.transformers_model import TransformersModelConfig
        from lighteval.models.transformers.vlm_transformers_model import VLMTransformersModelConfig
        from lighteval.models.vllm.vllm_model import VLLMModelConfig

        # Tested model configurations
        self.dummy_config = DummyModelConfig(model_name="test/case")
        self.endpoint_serverless_config = ServerlessEndpointModelConfig(model_name="test/case")
        self.endpoint_ie_config = InferenceEndpointModelConfig(model_name="test/case")
        self.endpoint_ip_config = InferenceProvidersModelConfig(model_name="test/case", provider="no_provider")
        self.endpoint_litellm_config = LiteLLMModelConfig(model_name="test/case")
        self.tgi_config = TGIModelConfig(model_name="test/case")
        self.sg_lang_config = SGLangModelConfig(model_name="test/case")
        self.transformers_config = TransformersModelConfig(model_name="test/case")
        self.vlm_transformers_config = VLMTransformersModelConfig(model_name="test/case")
        self.vllm_config = VLLMModelConfig(model_name="test/case")

        # Reference configurations for expected results
        ref_system_prompt = None
        ref_generation_parameters = {
            "num_blocks": None,
            "block_size": None,
            "early_stopping": None,
            "repetition_penalty": None,
            "frequency_penalty": None,
            "length_penalty": None,
            "presence_penalty": None,
            "max_new_tokens": None,
            "min_new_tokens": None,
            "seed": None,
            "stop_tokens": None,
            "temperature": 0,
            "top_k": None,
            "min_p": None,
            "top_p": None,
            "truncate_prompt": None,
            "cache_implementation": None,
            "response_format": None,
        }  # ruff: noqa: E501
        self.dummy_ref_config = {
            "model_name": "test/case",
            "seed": 42,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.endpoint_serverless_ref_config = {
            "model_name": "test/case",
            "add_special_tokens": True,
            "batch_size": 1,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.endpoint_ie_ref_config = {
            "endpoint_name": None,
            "model_name": "test/case",
            "reuse_existing": False,
            "accelerator": "gpu",
            "dtype": None,
            "vendor": "aws",
            "region": "us-east-1",
            "instance_size": None,
            "instance_type": None,
            "framework": "pytorch",
            "endpoint_type": "protected",
            "add_special_tokens": True,
            "revision": "main",
            "namespace": None,
            "image_url": None,
            "env_vars": None,
            "batch_size": 1,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.endpoint_ip_ref_config = {
            "model_name": "test/case",
            "provider": "no_provider",
            "timeout": None,
            "proxies": None,
            "org_to_bill": None,
            "parallel_calls_count": 10,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.endpoint_litellm_ref_config = {
            "model_name": "test/case",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.tgi_ref_config = {
            "inference_server_address": None,
            "inference_server_auth": None,
            "model_name": "test/case",
            "model_info": None,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.sg_lang_ref_config = {
            "model_name": "test/case",
            "load_format": "auto",
            "dtype": "auto",
            "tp_size": 1,
            "dp_size": 1,
            "context_length": None,
            "random_seed": 1234,
            "trust_remote_code": False,
            "device": "cuda",
            "skip_tokenizer_init": False,
            "kv_cache_dtype": "auto",
            "add_special_tokens": True,
            "pairwise_tokenization": False,
            "sampling_backend": None,
            "attention_backend": None,
            "mem_fraction_static": 0.8,
            "chunked_prefill_size": 4096,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.transformers_ref_config = {
            "model_name": "test/case",
            "tokenizer": None,
            "subfolder": None,
            "revision": "main",
            "batch_size": None,
            "max_length": None,
            "model_loading_kwargs": {},
            "add_special_tokens": True,
            "skip_special_tokens": True,
            "model_parallel": None,
            "dtype": None,
            "device": "cuda",
            "trust_remote_code": False,
            "compile": False,
            "multichoice_continuations_start_space": None,
            "pairwise_tokenization": False,
            "continuous_batching": False,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.vlm_transformers_ref_config = {
            "model_name": "test/case",
            "processor": None,
            "use_fast_image_processor": None,
            "subfolder": None,
            "revision": "main",
            "batch_size": 1,
            "generation_size": None,
            "max_length": None,
            "add_special_tokens": True,
            "model_parallel": None,
            "dtype": None,
            "device": "cuda",
            "trust_remote_code": False,
            "compile": False,
            "device_map": None,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501
        self.vllm_ref_config = {
            "model_name": "test/case",
            "revision": "main",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_length": None,
            "quantization": None,
            "load_format": None,
            "swap_space": 4,
            "seed": 1234,
            "trust_remote_code": False,
            "add_special_tokens": True,
            "multichoice_continuations_start_space": True,
            "pairwise_tokenization": False,
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048,
            "subfolder": None,
            "is_async": False,
            "system_prompt": ref_system_prompt,
            "generation_parameters": ref_generation_parameters,
        }  # ruff: noqa: E501

    def test_default_property_with_different_model_configs(self):
        """Test that results property correctly handles different model configurations."""
        for model_config in [
            self.dummy_config,
            self.endpoint_serverless_config,
            self.endpoint_ie_config,
            self.endpoint_ip_config,
            self.endpoint_litellm_config,
            self.tgi_config,
            self.sg_lang_config,
            self.transformers_config,
            self.vlm_transformers_config,
            self.vllm_config,
        ]:
            with self.subTest(model_config=model_config):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    evaluation_tracker = EvaluationTracker(output_dir=tmp_dir)

                evaluation_tracker.general_config_logger.log_model_info(
                    model_config=model_config,
                )

                results = evaluation_tracker.results

                # Verify structure
                self.assertIsInstance(results, dict)
                for key in [
                    "config_general",
                    "results",
                    "versions",
                    "config_tasks",
                    "summary_tasks",
                    "summary_general",
                ]:
                    self.assertIn(key, results.keys())

                # Configs should all be empty since not initialized
                self.assertEqual(results["versions"], {})
                self.assertEqual(results["config_tasks"], {})
                self.assertEqual(results["summary_tasks"], {})
                self.assertEqual(
                    results["summary_general"],
                    {
                        "hashes": {},
                        "truncated": 0,
                        "non_truncated": 0,
                        "padded": 0,
                        "non_padded": 0,
                    },
                )

                # Except config_general, which should contain the model config among other things
                general_config = results["config_general"]
                # We skip testing lighteval_sha, start_time
                self.assertIsNone(general_config["num_fewshot_seeds"])
                self.assertIsNone(general_config["max_samples"])
                self.assertIsNone(general_config["job_id"])
                self.assertIsNone(general_config["end_time"])
                self.assertIsNone(general_config["total_evaluation_time_secondes"])
                self.assertEqual(general_config["model_name"], "test/case")

    def test_model_config_property_with_different_model_configs(self):
        """Test that the model configs are properly saved."""
        for model_config, ref_config in [
            (self.dummy_config, self.dummy_ref_config),
            (self.endpoint_serverless_config, self.endpoint_serverless_ref_config),
            (self.endpoint_ie_config, self.endpoint_ie_ref_config),
            (self.endpoint_ip_config, self.endpoint_ip_ref_config),
            (self.endpoint_litellm_config, self.endpoint_litellm_ref_config),
            (self.tgi_config, self.tgi_ref_config),
            (self.sg_lang_config, self.sg_lang_ref_config),
            (self.transformers_config, self.transformers_ref_config),
            (self.vlm_transformers_config, self.vlm_transformers_ref_config),
            (self.vllm_config, self.vllm_ref_config),
        ]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                evaluation_tracker = EvaluationTracker(output_dir=tmp_dir)

                evaluation_tracker.general_config_logger.log_model_info(
                    model_config=model_config,
                )

                results = evaluation_tracker.results

                # Now to the core test, the model_config
                for k, v in ref_config.items():
                    with self.subTest(model_config=model_config, model_property=k):
                        self.assertEqual(results["config_general"]["model_config"][k], v)

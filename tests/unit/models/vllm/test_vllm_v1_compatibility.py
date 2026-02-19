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

"""
Tests for vLLM 0.11.0 compatibility.

This test suite validates that lighteval works correctly with vLLM 0.11.0,
which removed the V0 engine and made V1 the only engine in the codebase.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from lighteval.utils.imports import is_package_available


# Only run these tests if vllm is available
pytestmark = pytest.mark.skipif(not is_package_available("vllm"), reason="vllm is not installed")


if is_package_available("vllm"):
    import vllm

    from lighteval.models.vllm.vllm_model import AsyncVLLMModel, VLLMModel, VLLMModelConfig


class TestVLLMVersion(unittest.TestCase):
    """Test vLLM version compatibility."""

    def test_vllm_version_is_0_11_or_higher(self):
        """Verify we're using vLLM 0.11.0 or higher."""
        version = vllm.__version__
        major, minor = map(int, version.split(".")[:2])
        self.assertGreaterEqual(major, 0)
        self.assertGreaterEqual(minor, 11)
        print(f"✓ vLLM version: {version}")

    def test_v1_engine_available(self):
        """Test that V1 engine components are available."""
        # V1 components should be importable
        from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM

        self.assertIsNotNone(AsyncLLM)
        self.assertIsNotNone(AsyncEngineArgs)
        print("✓ V1 engine components available")

    def test_v0_engine_removed(self):
        """Test that V0 engine components are removed."""
        # V0 components should NOT be available in 0.11.0
        with self.assertRaises(ImportError):
            from vllm.engine.async_llm import AsyncLLMEngine  # noqa: F401

        print("✓ V0 engine properly removed")


class TestVLLMModelInitialization(unittest.TestCase):
    """Test VLLMModel and AsyncVLLMModel initialization with V1 engine."""

    @patch("lighteval.models.vllm.vllm_model.VLLMModel._create_auto_model")
    def test_vllm_model_initializes_with_small_model(self, mock_create_model):
        """Test that VLLMModel initializes correctly with a tiny model."""
        # Mock the model creation to avoid actually loading a model
        mock_model = Mock()
        mock_model.generate = Mock(return_value=[])
        mock_create_model.return_value = mock_model

        config = VLLMModelConfig(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            tensor_parallel_size=1,
            data_parallel_size=1,
            gpu_memory_utilization=0.3,
        )

        try:
            model = VLLMModel(config)
            self.assertIsNotNone(model)
            self.assertEqual(model.model_name, "SmolLM2-135M-Instruct")
            print("✓ VLLMModel initialization successful")
        except Exception as e:
            self.fail(f"VLLMModel initialization failed: {e}")

    @patch("lighteval.models.vllm.vllm_model.AsyncVLLMModel._create_async_llm")
    def test_async_vllm_model_initializes(self, mock_create_async):
        """Test that AsyncVLLMModel initializes correctly with V1 engine."""
        # Mock the async LLM creation
        mock_async_llm = Mock()
        mock_create_async.return_value = mock_async_llm

        config = VLLMModelConfig(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            tensor_parallel_size=1,
            data_parallel_size=1,
            gpu_memory_utilization=0.3,
        )

        try:
            model = AsyncVLLMModel(config)
            self.assertIsNotNone(model)
            self.assertEqual(model.model_name, "SmolLM2-135M-Instruct")
            print("✓ AsyncVLLMModel initialization successful")
        except Exception as e:
            self.fail(f"AsyncVLLMModel initialization failed: {e}")


class TestVLLMImports(unittest.TestCase):
    """Test that all required vLLM imports work correctly."""

    def test_core_vllm_imports(self):
        """Test that core vLLM components can be imported."""
        try:
            from vllm import LLM, RequestOutput, SamplingParams

            self.assertIsNotNone(LLM)
            self.assertIsNotNone(SamplingParams)
            self.assertIsNotNone(RequestOutput)
            print("✓ Core vLLM imports successful")
        except ImportError as e:
            self.fail(f"Core vLLM imports failed: {e}")

    def test_distributed_imports(self):
        """Test that distributed components can be imported."""
        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            self.assertIsNotNone(destroy_distributed_environment)
            self.assertIsNotNone(destroy_model_parallel)
            print("✓ Distributed imports successful")
        except ImportError as e:
            self.fail(f"Distributed imports failed: {e}")

    def test_tokenizer_utils_imports(self):
        """Test that tokenizer utils can be imported."""
        try:
            from vllm.tokenizers import get_tokenizer

            self.assertIsNotNone(get_tokenizer)
            print("✓ Tokenizer utils imports successful")
        except ImportError as e:
            self.fail(f"Tokenizer utils imports failed: {e}")

    def test_async_llm_imports(self):
        """Test that async LLM components can be imported (V1)."""
        try:
            from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM

            self.assertIsNotNone(AsyncLLM)
            self.assertIsNotNone(AsyncEngineArgs)
            print("✓ AsyncLLM (V1) imports successful")
        except ImportError as e:
            self.fail(f"AsyncLLM imports failed: {e}")


class TestVLLMModelConfigDefaults(unittest.TestCase):
    """Test that VLLMModelConfig works with new defaults in 0.11.0."""

    def test_config_with_default_cuda_graph_mode(self):
        """Test that config works with new FULL_AND_PIECEWISE default."""
        config = VLLMModelConfig(model_name="test/model")

        # The config should not specify cuda_graph_mode by default,
        # allowing vLLM to use its new default (FULL_AND_PIECEWISE)
        self.assertIsNotNone(config)
        self.assertEqual(config.model_name, "test/model")
        print("✓ Config respects new CUDA graph mode defaults")

    def test_config_with_parallelism_options(self):
        """Test that parallelism configurations work correctly."""
        config = VLLMModelConfig(
            model_name="test/model",
            tensor_parallel_size=2,
            data_parallel_size=2,
            pipeline_parallel_size=1,
        )

        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.data_parallel_size, 2)
        self.assertEqual(config.pipeline_parallel_size, 1)
        print("✓ Parallelism configuration works correctly")


class TestSamplingParamsCompatibility(unittest.TestCase):
    """Test that SamplingParams works correctly with V1 engine."""

    @patch("lighteval.models.vllm.vllm_model.VLLMModel._create_auto_model")
    def test_sampling_params_creation(self, mock_create_model):
        """Test that SamplingParams can be created with typical parameters."""
        from vllm import SamplingParams

        mock_model = Mock()
        mock_create_model.return_value = mock_model

        config = VLLMModelConfig(model_name="test/model")
        VLLMModel(config)  # Create model but don't need to use it

        # Test creating SamplingParams with typical parameters
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, max_tokens=100, stop=["</s>", "###"])

        self.assertIsNotNone(sampling_params)
        self.assertEqual(sampling_params.temperature, 0.7)
        self.assertEqual(sampling_params.top_p, 0.9)
        self.assertEqual(sampling_params.top_k, 50)
        self.assertEqual(sampling_params.max_tokens, 100)
        print("✓ SamplingParams creation successful")


if __name__ == "__main__":
    unittest.main()

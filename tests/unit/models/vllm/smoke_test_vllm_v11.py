#!/usr/bin/env python3
# MIT License

# Copyright (c) 2025 The HuggingFace Team

"""
Quick smoke tests for vLLM 0.11.0 compatibility.

This script performs basic sanity checks to verify that vLLM 0.11.0
works with lighteval's integration.
"""

import sys


def test_vllm_import():
    """Test basic vLLM import."""
    print("Testing vLLM import...")
    try:
        import vllm

        print(f"✓ vLLM imported successfully. Version: {vllm.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import vLLM: {e}")
        return False


def test_vllm_version():
    """Test vLLM version is 0.11.0 or higher."""
    print("\nTesting vLLM version...")
    try:
        import vllm

        version = vllm.__version__
        major, minor = map(int, version.split(".")[:2])

        if major > 0 or (major == 0 and minor >= 11):
            print(f"✓ vLLM version {version} is 0.11.0 or higher")
            return True
        else:
            print(f"✗ vLLM version {version} is lower than 0.11.0")
            return False
    except Exception as e:
        print(f"✗ Failed to check vLLM version: {e}")
        return False


def test_v1_engine_imports():
    """Test V1 engine imports."""
    print("\nTesting V1 engine imports...")
    try:
        from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM  # noqa: F401

        print("✓ V1 AsyncLLM engine imports successful")
        return True
    except ImportError as e:
        print(f"✗ Failed to import V1 engine: {e}")
        return False


def test_v0_engine_removed():
    """Test that V0 engine is removed."""
    print("\nTesting V0 engine removal...")
    try:
        from vllm.engine.async_llm import AsyncLLMEngine  # noqa: F401

        print("✗ V0 engine still present (should be removed in 0.11.0)")
        return False
    except ImportError:
        print("✓ V0 engine properly removed")
        return True


def test_core_imports():
    """Test core vLLM component imports."""
    print("\nTesting core vLLM imports...")
    try:
        from vllm import LLM, RequestOutput, SamplingParams  # noqa: F401
        from vllm.distributed.parallel_state import (  # noqa: F401
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm.tokenizers import get_tokenizer  # noqa: F401

        print("✓ All core vLLM components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import core components: {e}")
        return False


def test_lighteval_vllm_imports():
    """Test lighteval's vLLM integration imports."""
    print("\nTesting lighteval vLLM integration imports...")
    try:
        from lighteval.models.vllm.vllm_model import AsyncVLLMModel, VLLMModel, VLLMModelConfig  # noqa: F401

        print("✓ Lighteval vLLM integration imports successful")
        return True
    except ImportError as e:
        print(f"✗ Failed to import lighteval vLLM integration: {e}")
        return False


def test_model_config_creation():
    """Test VLLMModelConfig creation."""
    print("\nTesting VLLMModelConfig creation...")
    try:
        from lighteval.models.vllm.vllm_model import VLLMModelConfig

        config = VLLMModelConfig(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            tensor_parallel_size=1,
            data_parallel_size=1,
            gpu_memory_utilization=0.3,
        )

        print("✓ VLLMModelConfig created successfully")
        print(f"  Model: {config.model_name}")
        print(f"  TP: {config.tensor_parallel_size}, DP: {config.data_parallel_size}")
        return True
    except Exception as e:
        print(f"✗ Failed to create VLLMModelConfig: {e}")
        return False


def test_sampling_params():
    """Test SamplingParams creation."""
    print("\nTesting SamplingParams creation...")
    try:
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, max_tokens=100, stop=["</s>"])

        print("✓ SamplingParams created successfully")
        print(f"  Temperature: {params.temperature}")
        print(f"  Top-p: {params.top_p}")
        print(f"  Max tokens: {params.max_tokens}")
        return True
    except Exception as e:
        print(f"✗ Failed to create SamplingParams: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("vLLM 0.11.0 Compatibility Smoke Tests")
    print("=" * 60)

    tests = [
        test_vllm_import,
        test_vllm_version,
        test_v1_engine_imports,
        test_v0_engine_removed,
        test_core_imports,
        test_lighteval_vllm_imports,
        test_model_config_creation,
        test_sampling_params,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All smoke tests passed!")
        print("=" * 60)
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

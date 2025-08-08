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

"""
Tests for the caching infrastructure in VLLMModel.
"""

import tempfile


# Test imports
try:
    from lighteval.models.utils import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
    from lighteval.tasks.requests import Doc
    from lighteval.utils.cache_management import SampleCache

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def test_cache_initialization():
    """Test that cache is properly initialized in VLLMModel."""
    config = VLLMModelConfig(
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision="main",
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        max_model_length=512,  # Small for testing
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        generation_parameters=GenerationParameters(
            temperature=0.0,
            max_new_tokens=10,  # Very small for testing
        ),
    )

    # Test cache initialization
    cache = SampleCache(config)

    assert cache.cache_dir is not None
    assert cache.model_hash is not None
    assert cache.tokenization_dir.exists()
    assert cache.predictions_dir.exists()


def test_cache_decorator_presence():
    """Test that @cached decorators are present on the right methods."""
    # Check if the methods have the cached decorator
    methods_to_check = ["greedy_until", "loglikelihood", "loglikelihood_rolling"]

    for method_name in methods_to_check:
        assert hasattr(VLLMModel, method_name), f"{method_name} method not found"
        method = getattr(VLLMModel, method_name)
        # Check if method has been wrapped by @cached decorator
        assert hasattr(method, "__wrapped__"), f"{method_name} missing @cached decorator"


def test_sample_docs():
    """Create sample Doc objects for testing."""
    docs = [
        Doc(
            id="doc1",
            task_name="test_task",
            query="What is the capital of France?",
            choices=["Paris", "London", "Berlin", "Madrid"],
            gold_index=0,
            instruction="Answer the question",
        ),
        Doc(
            id="doc2",
            task_name="test_task",
            query="What color is the sky?",
            choices=["Blue", "Red", "Green", "Yellow"],
            gold_index=0,
            instruction="Answer the question",
        ),
    ]
    assert len(docs) == 2
    assert all(doc.task_name == "test_task" for doc in docs)
    assert all(len(doc.choices) == 4 for doc in docs)


def test_cache_directory_structure():
    """Test that cache directories are created correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = VLLMModelConfig(
            model_name="test-model", revision="main", dtype="float16", generation_parameters=GenerationParameters()
        )

        # Create cache with custom directory
        cache = SampleCache(config, cache_dir=temp_dir)

        # Check directory structure
        assert cache.tokenization_dir.exists()
        assert cache.predictions_dir.exists()
        assert str(temp_dir) in str(cache.cache_dir)


# These tests can be run with pytest as part of the test suite

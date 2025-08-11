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
Integration tests to verify that caching actually works with real model predictions.
"""

import tempfile
import time
import unittest

from lighteval.tasks.requests import Doc


class TestCaching(unittest.TestCase):
    def setUp(self):
        """Create simple test documents."""
        self.docs = []
        for i in range(3):
            doc = Doc(
                id=f"test_doc_{i}",
                task_name="cache_test",
                query=f"Test question {i}: What is 2+2?",
                choices=["3", "4", "5", "6"],
                gold_index=1,  # "4" is correct
                instruction="Answer the math question",
                fewshot_sorting_class="math_questions",
            )
            self.docs.append(doc)

    def test_cache_with_mock_model(self):
        """Test caching with a mock model that doesn't require GPU."""
        from lighteval.models.model_input import GenerationParameters
        from lighteval.models.model_output import ModelResponse
        from lighteval.models.vllm.vllm_model import VLLMModelConfig
        from lighteval.utils.cache_management import SampleCache

        # Create a temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config
            config = VLLMModelConfig(
                model_name="test-model",
                revision="main",
                dtype="float16",
                generation_parameters=GenerationParameters(temperature=0.0, max_new_tokens=10),
            )

            # Create cache
            cache = SampleCache(config, cache_dir=temp_dir)
            task_name = "cache_test"

            # Test storing predictions
            mock_responses = []
            for i, doc in enumerate(self.docs):
                response = ModelResponse(
                    input=doc.query,
                    text=[f"Answer {i}"],
                    input_tokens=[1, 2, 3, 4],
                    output_tokens=[[5, 6, 7]],
                    logprobs=[-0.1, -0.2, -0.3],
                    argmax_logits_eq_gold=True,
                )
                mock_responses.append(response)

            # Store in cache
            cache.store_predictions(self.docs, mock_responses, task_name)

            # Verify cache files were created
            cache_file = cache.predictions_dir / f"{task_name}.parquet"
            assert cache_file.exists(), "Cache file not created"

            # Test retrieving from cache
            cached_results, uncached_docs = cache.get_predictions(self.docs, task_name)

            # Verify all docs were cached
            assert len(uncached_docs) == 0, f"{len(uncached_docs)} documents not found in cache"

            # Verify cached results match original
            for i, (original, cached) in enumerate(zip(mock_responses, cached_results)):
                assert cached is not None, f"Document {i} not found in cache"
                assert cached.text == original.text, f"Document {i} text mismatch: {cached.text} != {original.text}"

            # Test cache hit detection
            has_predictions = [cache.has_prediction(doc, task_name) for doc in self.docs]
            assert all(has_predictions), "Cache hit detection failed"

    def test_cache_decorator_simulation(self):
        """Test that the @cached decorator logic works correctly."""
        from lighteval.models.model_input import GenerationParameters
        from lighteval.models.model_output import ModelResponse
        from lighteval.models.vllm.vllm_model import VLLMModelConfig
        from lighteval.utils.cache_management import SampleCache, cached

        # Create a mock model class to test the decorator
        class MockModel:
            def __init__(self, cache_dir):
                config = VLLMModelConfig(
                    model_name="mock-model",
                    revision="main",
                    dtype="float16",
                    generation_parameters=GenerationParameters(),
                )
                self._config = config
                self._cache = SampleCache(config, cache_dir=cache_dir)
                self.call_count = 0

            @cached("predictions")
            def mock_prediction_method(self, docs: list[Doc]) -> list[ModelResponse]:
                """Mock method that simulates a prediction."""
                self.call_count += 1

                # Simulate some processing time
                time.sleep(0.1)

                # Return mock responses
                responses = []
                for i, doc in enumerate(docs):
                    response = ModelResponse(
                        input=doc.query,
                        text=[f"Mock response for {doc.query}"],
                        input_tokens=[1, 2, 3],
                        output_tokens=[[4, 5, 6]],
                    )
                    responses.append(response)
                return responses

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock model
            model = MockModel(temp_dir)

            # First call - should execute the method
            results1 = model.mock_prediction_method(self.docs)

            assert model.call_count == 1, f"Expected 1 call, got {model.call_count}"

            # Second call - should use cache
            results2 = model.mock_prediction_method(self.docs)

            assert model.call_count == 1, f"Expected 1 call after cache hit, got {model.call_count}"

            # Verify results are the same
            assert len(results1) == len(results2), "Results length mismatch"

            for r1, r2 in zip(results1, results2):
                assert r1.text == r2.text, "Results content mismatch"

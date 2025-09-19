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
from pathlib import Path
from types import ModuleType
from typing import Optional, Union
from unittest.mock import patch

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.imports import is_package_available


class TestPipelineReasoningTags(unittest.TestCase):
    """Test suite for pipeline reasoning tags functionality using DummyModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple test task
        self.task_config = LightevalTaskConfig(
            name="test_reasoning_task",
            suite=["test"],
            prompt_function=lambda x: x,
            hf_repo="test_repo",
            hf_subset="default",
            metrics=[Metrics.exact_match],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=10,
            stop_sequence=["\n"],
            num_fewshots=0,
        )
        self.input_task_name = "test|test_reasoning_task|0"
        self.task_config_name = self.task_config.full_name

        # Create test documents with reasoning tags in expected responses
        self.test_docs = [
            Doc(
                task_name=self.input_task_name,
                query="What is 2+2?",
                choices=["4"],
                gold_index=[0],
                instruction="",
                sampling_methods=[SamplingMethod.GENERATIVE],
            ),
        ]

        # Mock dataset
        self.mock_dataset = {"test": self.test_docs}

    def _mock_task_registry(self, input_task_name, task_config, task_docs, responses_with_reasoning_tags):
        """Create a fake registry for testing."""

        class FakeTask(LightevalTask):
            def __post_init__(self):
                self._docs = task_docs

            def get_docs(self, max_samples=None):
                return task_docs

            @staticmethod
            def download_dataset_worker(task) -> None:
                # Mock dataset loading
                return task._docs

        class FakeRegistry(Registry):
            def __init__(
                self, tasks: Optional[str] = None, custom_tasks: Optional[Union[str, Path, ModuleType]] = None
            ):
                self.tasks_list = [input_task_name]
                # suite_name, task_name, few_shot = input_task_name.split("|")
                self.task_to_configs = {input_task_name: [task_config]}

            def load_tasks(self):
                return {input_task_name: FakeTask(config=task_config)}

        # Create a DummyModel that returns responses with reasoning tags
        class TestDummyModel(DummyModel):
            def __init__(self, config):
                super().__init__(config)

            def greedy_until(self, docs):
                # Return responses with reasoning tags
                return responses_with_reasoning_tags

        return FakeRegistry, TestDummyModel

    def test_remove_reasoning_tags_enabled(self):
        """Test that reasoning tags are removed when remove_reasoning_tags=True."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal enabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags=[("<think>", "</think>")],
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test|test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test|test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["The answer is 4"]
            )

    def test_remove_reasoning_tags_enabled_tags_as_string(self):
        """Test that reasoning tags are removed when remove_reasoning_tags=True."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal enabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags='[("<think>", "</think>")]',
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test|test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test|test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["The answer is 4"]
            )

    def test_remove_reasoning_tags_enabled_default_tags(self):
        """Test that reasoning tags are removed when remove_reasoning_tags=True."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal enabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE, remove_reasoning_tags=True, max_samples=1
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test|test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test|test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["The answer is 4"]
            )

    def test_remove_reasoning_tags_disabled(self):
        """Test that reasoning tags are preserved when remove_reasoning_tags=False."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal disabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=False,
                reasoning_tags=[("<think>", "</think>")],
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test|test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that post-processed text is None (= no post processing happened)
            details = pipeline.evaluation_tracker.details
            self.assertIsNone(
                details["test|test_reasoning_task|0"][0]["model_response"]["text_post_processed"],
            )

    def test_custom_reasoning_tags(self):
        """Test that custom reasoning tags are correctly applied."""

        # Responses with custom reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["[reasoning]This is my thought process[/reasoning]Final answer: 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with custom reasoning tags
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags=[("[reasoning]", "[/reasoning]")],
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test|test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test|test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["Final answer: 4"]
            )

    def test_multiple_reasoning_tags(self):
        """Test that multiple reasoning tag pairs are correctly handled."""

        # Responses with multiple reasoning tag types
        responses_with_reasoning = [
            ModelResponse(text=["<think>First thought</think>Some text<reason>Second thought</reason>Final: 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with multiple reasoning tag pairs
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags='[("<think>", "</think>"), ("<reason>", "</reason>")]',
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test|test|test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test|test_reasoning_task|0"][0]["model_response"]["text_post_processed"],
                ["Some textFinal: 4"],
            )

    def test_reasoning_tags_validation(self):
        """Test that invalid reasoning_tags parameter raises appropriate error."""

        for test_string in ["['incorrect_format']", "invalid_format"]:
            with self.assertRaises(ValueError) as context:
                PipelineParameters(
                    launcher_type=ParallelismManager.NONE,
                    reasoning_tags=test_string,  # Should be a list of tuples
                )

            # Check that the error message mentions the expected format
            print(context.__dict__)
            self.assertIn("reasoning_tags must be a list of pair tuples", str(context.exception))

    def test_default_reasoning_tags(self):
        """Test that default reasoning tags are correctly set."""

        pipeline_params = PipelineParameters(launcher_type=ParallelismManager.NONE)

        # Check that default reasoning tags are set
        self.assertEqual(pipeline_params.reasoning_tags, [("<think>", "</think>")])
        self.assertTrue(pipeline_params.remove_reasoning_tags)


if __name__ == "__main__":
    unittest.main()

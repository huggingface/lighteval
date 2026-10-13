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
from typing import Optional
from unittest.mock import patch

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod


class TestLoadResponsesFromDetails(unittest.TestCase):
    """Test suite for reloading model responses from previously saved details."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_names = ["task_a|0", "task_b|0"]

        self.docs = {
            task_name: [
                Doc(
                    task_name=task_name,
                    query=f"{task_name} query {i}",
                    choices=["4"],
                    gold_index=[0],
                    instruction="",
                    sampling_methods=[SamplingMethod.GENERATIVE],
                )
                for i in range(2)
            ]
            for task_name in self.task_names
        }

    def _task_config(self, task_name: str) -> LightevalTaskConfig:
        return LightevalTaskConfig(
            name=task_name.split("|")[0],
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

    def _build_pipeline(self) -> Pipeline:
        docs_per_task = self.docs

        class FakeTask(LightevalTask):
            def __init__(self, config, task_name):
                super().__init__(config=config)
                self._docs = docs_per_task[task_name]

            def get_docs(self, max_samples=None):
                return self._docs

        class FakeRegistry(Registry):
            def __init__(
                self, tasks: Optional[str] = None, load_multilingual: bool = False, custom_tasks: Optional[str] = None
            ):
                self.tasks_list = list(docs_per_task.keys())

            def load_tasks(inner_self):
                return {
                    task_name: FakeTask(config=self._task_config(task_name), task_name=task_name)
                    for task_name in docs_per_task
                }

        with patch("lighteval.pipeline.Registry", FakeRegistry), patch.object(LightevalTask, "load_datasets"):
            return Pipeline(
                tasks="|".join(self.task_names),
                pipeline_parameters=PipelineParameters(
                    launcher_type=ParallelismManager.NONE,
                    load_responses_from_details_date_id="2024-01-01T00-00-00.000000",
                ),
                evaluation_tracker=EvaluationTracker(output_dir=self.temp_dir),
                model=DummyModel(DummyModelConfig(seed=42)),
            )

    def test_responses_of_all_tasks_are_loaded(self):
        """All tasks' responses are reloaded, in the same order as the pipeline documents."""
        pipeline = self._build_pipeline()

        details_datasets = {
            task_name: [
                {"model_response": {"text": [f"{task_name} prediction {i}"]}} for i in range(len(self.docs[task_name]))
            ]
            for task_name in self.task_names
        }

        with patch.object(pipeline.evaluation_tracker, "load_details_datasets", return_value=details_datasets):
            model_responses = pipeline._load_responses_from_details()

        expected = [
            f"{task_name} prediction {i}" for task_name in self.task_names for i in range(len(self.docs[task_name]))
        ]
        assert [response.text[0] for response in model_responses[SamplingMethod.GENERATIVE]] == expected

    def test_responses_follow_the_document_order(self):
        """Details are realigned on the pipeline task order, whatever order they were loaded in."""
        pipeline = self._build_pipeline()

        details_datasets = {
            task_name: [
                {"model_response": {"text": [f"{task_name} prediction {i}"]}} for i in range(len(self.docs[task_name]))
            ]
            for task_name in reversed(self.task_names)
        }

        with patch.object(pipeline.evaluation_tracker, "load_details_datasets", return_value=details_datasets):
            model_responses = pipeline._load_responses_from_details()

        loaded_texts = [response.text[0] for response in model_responses[SamplingMethod.GENERATIVE]]
        assert loaded_texts == [
            f"{task_name} prediction {i}" for task_name in self.task_names for i in range(len(self.docs[task_name]))
        ]


if __name__ == "__main__":
    unittest.main()

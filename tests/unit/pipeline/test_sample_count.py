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
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import make_results_table


def test_sample_count_uses_filtered_evaluation_docs():
    config = LightevalTaskConfig(
        name="filtered_task",
        prompt_function=lambda line, task_name=None: Doc(query=line["question"], choices=[""], gold_index=0),
        hf_repo="unused",
        hf_subset="default",
        metrics=[Metrics.exact_match],
        hf_filter=lambda line: line["keep"],
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
    )
    task = LightevalTask(config)
    raw_dataset = DatasetDict(
        {
            "test": Dataset.from_list(
                [
                    {"question": "kept 1", "keep": True},
                    {"question": "removed", "keep": False},
                    {"question": "kept 2", "keep": True},
                ]
            )
        }
    )

    class FakeRegistry(Registry):
        def __init__(self, tasks, load_multilingual=False, custom_tasks=None):
            self.tasks_list = [config.full_name]

        def load_tasks(self):
            return {config.full_name: task}

    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = EvaluationTracker(output_dir=temp_dir)
        with (
            patch("lighteval.pipeline.Registry", FakeRegistry),
            patch("lighteval.tasks.lighteval_task.load_dataset", return_value=raw_dataset),
        ):
            Pipeline(
                tasks=config.full_name,
                pipeline_parameters=PipelineParameters(launcher_type=ParallelismManager.NONE),
                evaluation_tracker=tracker,
                model=DummyModel(DummyModelConfig()),
            )

        tracker.metrics_logger.metric_aggregated = {config.full_name: {"accuracy": 0.0}}
        tracker.versions_logger.versions = {config.full_name: 1}
        table = make_results_table(tracker.generate_final_dict())

        row = next(row for row in table.splitlines() if "filtered_task:0" in row)
        cells = row.strip("|").split("|")
        assert [cell.strip() for cell in cells[:4]] == ["filtered_task:0", "1", "2", "accuracy"]

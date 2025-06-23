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

from pathlib import Path
from types import ModuleType
from typing import Optional, Union
from unittest.mock import patch

from transformers import AutoTokenizer

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_accelerate_available


class FakeModel(LightevalModel):
    """Fake model for testing purposes."""

    def __init__(
        self,
        greedy_until_responses: list[ModelResponse] = [],
        loglikelihood_responses: list[ModelResponse] = [],
        loglikelihood_rolling_responses: list[ModelResponse] = [],
    ):
        self._tokenizer = None
        self.greedy_until_responses = greedy_until_responses
        self.loglikelihood_responses = loglikelihood_responses
        self.loglikelihood_rolling_responses = loglikelihood_rolling_responses

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def model_info(self):
        return ModelInfo(model_name="fake_model")

    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        ret_resp, self.greedy_until_responses = (
            self.greedy_until_responses[: len(docs)],
            self.greedy_until_responses[len(docs) :],
        )
        return ret_resp

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        ret_resp, self.loglikelihood_responses = (
            self.loglikelihood_responses[: len(docs)],
            self.loglikelihood_responses[len(docs) :],
        )
        return ret_resp

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        ret_resp, self.loglikelihood_rolling_responses = (
            self.loglikelihood_rolling_responses[: len(docs)],
            self.loglikelihood_rolling_responses[len(docs) :],
        )
        return ret_resp


def fake_evaluate_task(
    lighteval_task: LightevalTask, lm: FakeModel, max_samples: int = 1, n_fewshot: int = 0, n_fewshot_seeds: int = 1
):
    # Mock the Registry.get_task_dict method

    task_name = f"{lighteval_task.suite[0]}|{lighteval_task.name}"

    task_dict = {task_name: lighteval_task}
    evaluation_tracker = EvaluationTracker(output_dir="outputs")
    evaluation_tracker.task_config_logger.log(task_dict)
    # Create a mock Registry class

    class FakeRegistry(Registry):
        def __init__(self, custom_tasks: Optional[Union[str, Path, ModuleType]] = None):
            super().__init__(custom_tasks=custom_tasks)

        def get_task_dict(self, task_names: list[str]):
            return task_dict

        def get_tasks_configs(self, task: str):
            config = lighteval_task.config
            config.num_fewshots = n_fewshot
            config.truncate_fewshots = False
            config.full_name = f"{task_name}|{config.num_fewshots}"
            return [config]

    # This is due to logger complaining we have no initialised the accelerator
    # It's hard to mock as it's global singleton
    if is_accelerate_available():
        from accelerate import Accelerator

        Accelerator()

    # This is a bit hacky, because there is no way to run end to end, with
    # dynamic task :(, so we just mock the registry
    with patch("lighteval.pipeline.Registry", FakeRegistry):
        pipeline = Pipeline(
            tasks=task_name,
            pipeline_parameters=PipelineParameters(max_samples=max_samples, launcher_type=ParallelismManager.NONE),
            evaluation_tracker=evaluation_tracker,
            model=lm,
            model_config=None,
        )
        pipeline.evaluate()

    return pipeline.get_results()

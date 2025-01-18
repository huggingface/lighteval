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
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.imports import is_accelerate_available


class FakeModel(LightevalModel):
    """Fake model for testing purposes."""

    def __init__(
        self,
        greedy_until_responses: list[GenerativeResponse] = [],
        loglikelihood_responses: list[LoglikelihoodResponse] = [],
        loglikelihood_rolling_responses: list[LoglikelihoodResponse] = [],
        loglikelihood_single_token_responses: list[LoglikelihoodSingleTokenResponse] = [],
    ):
        self._tokenizer = None
        self.greedy_until_responses = greedy_until_responses
        self.loglikelihood_responses = loglikelihood_responses
        self.loglikelihood_rolling_responses = loglikelihood_rolling_responses
        self.loglikelihood_single_token_responses = loglikelihood_single_token_responses

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

    def greedy_until(
        self, requests: list[GreedyUntilRequest], override_bs: Optional[int] = None
    ) -> list[GenerativeResponse]:
        ret_resp, self.greedy_until_resp = (
            self.greedy_until_responses[: len(requests)],
            self.greedy_until_responses[len(requests) :],
        )
        return ret_resp

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        ret_resp, self.loglikelihood_responses = (
            self.loglikelihood_responses[: len(requests)],
            self.loglikelihood_responses[len(requests) :],
        )
        return ret_resp

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        ret_resp, self.loglikelihood_rolling_responses = (
            self.loglikelihood_rolling_responses[: len(requests)],
            self.loglikelihood_rolling_responses[len(requests) :],
        )
        return ret_resp

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        ret_resp, self.loglikelihood_single_token_responses = (
            self.loglikelihood_single_token_responses[: len(requests)],
            self.loglikelihood_single_token_responses[len(requests) :],
        )
        return ret_resp


def fake_evaluate_task(
    task: LightevalTask, lm: FakeModel, max_samples: int = 1, n_fewshot: int = 0, n_fewshot_seeds: int = 1
):
    # Mock the Registry.get_task_dict method

    task_name = f"{task.suite[0]}|{task.name}"

    task_dict = {task_name: task}
    evaluation_tracker = EvaluationTracker(output_dir="outputs")
    evaluation_tracker.task_config_logger.log(task_dict)
    # Create a mock Registry class

    class FakeRegistry(Registry):
        def __init__(
            self, cache_dir: Optional[str] = None, custom_tasks: Optional[Union[str, Path, ModuleType]] = None
        ):
            super().__init__(cache_dir=cache_dir, custom_tasks=custom_tasks)

        def get_task_dict(self, task_names: list[str]):
            return task_dict

    # This is due to logger complaining we have no initialised the accelerator
    # It's hard to mock as it's global singleton
    if is_accelerate_available():
        from accelerate import Accelerator

        Accelerator()

    # This is a bit hacky, because there is no way to run end to end, with
    # dynamic task :(, so we just mock the registry
    task_run_string = f"{task_name}|{n_fewshot}|{n_fewshot_seeds}"
    with patch("lighteval.pipeline.Registry", FakeRegistry):
        pipeline = Pipeline(
            tasks=task_run_string,
            pipeline_parameters=PipelineParameters(max_samples=max_samples, launcher_type=ParallelismManager.NONE),
            evaluation_tracker=evaluation_tracker,
            model=lm,
            model_config=None,
        )
        pipeline.evaluate()

    return evaluation_tracker.metrics_logger.metrics_values[f"{task_name}|{n_fewshot}"]

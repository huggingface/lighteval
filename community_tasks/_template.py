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

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# DEFINE YOUR PROMPT FUNCTIONS
# Define as many as you need for your different tasks
def prompt_fn(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """
    return Doc(
        task_name=task_name,
        query="",
        choices="",
        gold_index=0,
        instruction="",
    )


# EVAL WITH NO SUBSET ##
# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
task = LightevalTaskConfig(
    name="myothertask",
    prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="",
    hf_subset="default",
    hf_avail_splits=[],
    evaluation_splits=[],
    few_shots_split="",
    few_shots_select="",
    metric=[],  # select your metric in Metrics
)

# EVALS WITH SUBSET
# This is how you create a subset task (like MMLU), which has several subset
# each being its own evaluation task.

# fmt: off
SAMPLE_SUBSETS = []  # list of all the subsets to use for this eval
# fmt: on


class CustomSubsetTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
            hf_repo="",
            metric=[custom_metric],  # select your metric in Metrics or use your custom_metric
            hf_avail_splits=[],
            evaluation_splits=[],
            few_shots_split="",
            few_shots_select="",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
        )


# STORE YOUR EVALS
SUBSET_TASKS = [CustomSubsetTask(name=f"mytask:{subset}", hf_subset=subset) for subset in SAMPLE_SUBSETS]
TASKS_TABLE = SUBSET_TASKS + [task]


# CUSTOM METRIC IF NEEDED
custom_metric = SampleLevelMetric(
    metric_name="my_custom_metric_name",
    higher_is_better=True,
    category=MetricCategory.IGNORED,
    use_case=MetricUseCase.NONE,
    sample_level_fn=lambda x: x,  # how to compute score for one sample
    corpus_level_fn=np.mean,  # aggregation
)

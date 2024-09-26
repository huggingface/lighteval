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

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""
import random

import numpy as np

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# DEFINE YOUR PROMPT FUNCTIONS
# Define as many as you need for your different tasks
def nocha_prompt(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """
    instruction = "According to the previous text, is the following claim True or False?"
    choices_true = random.sample(["True", "False"], 2)
    choices_false = random.sample(["True", "False"], 2)
    return [
        Doc(
            task_name=task_name,
            query=f"{line['context']}\n{instruction}\nClaim: {line['true_claim']}\nAnswer: ",
            choices=choices_true,
            gold_index=choices_true.index("True"),
            instruction="",
        ),
        Doc(
            task_name=task_name,
            query=f"{line['context']}\n{instruction}\nClaim: {line['false_claim']}\nAnswer: ",
            choices=choices_false,
            gold_index=choices_false.index("False"),
            instruction="",
        ),
    ]


# EVAL WITH NO SUBSET ##
# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.


nocha_task = LightevalTaskConfig(
    name="nocha",
    prompt_function=nocha_prompt,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="novelchallenge/nocha",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="sequential",  # should not be possible to run in few shot
    metric=[Metrics.loglikelihood_acc],  # select your metric in Metrics
    trust_dataset=True,
)

TASKS_TABLE = [nocha_task]

# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))

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


def nocha_prompt(line, task_name: str = None):
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


nocha_task = LightevalTaskConfig(
    name="nocha",
    prompt_function=nocha_prompt,
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


def musr2_prompt(line, task_name: str = None):
    # choices = [choice.split("- ")[-1] for choice in line["choices"].split("\n")]
    return (
        Doc(
            task_name=task_name,
            query=f"{line['prompt']}\nQuestion: {line['question']}\nChoices: {line['choices']}\nAnswer: ",
            choices=["1", "2", "3"],
            gold_index=str(line["answer"]),
            instruction="",
        ),
    )


musr2 = LightevalTaskConfig(
    name="musr2",
    prompt_function=musr2_prompt,
    suite=["community"],
    hf_repo="lighteval/MuSR2",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="sequential",  # should not be possible to run in few shot
    metric=[Metrics.loglikelihood_acc],
    trust_dataset=True,
)

TASKS_TABLE = [nocha_task, musr2]

# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))

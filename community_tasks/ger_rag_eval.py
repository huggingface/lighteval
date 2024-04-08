# MIT License

# Copyright (c) 2024 The HuggingFace Team
# Copyright (c) 2024 Philip May, Deutsche Telekom AG

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
import numpy as np
from aenum import extend_enum

from lighteval.metrics import Metrics
from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES


task1 = LightevalTaskConfig(
    name="ger_rag_eval_task1",  # ok
    prompt_function="prompt_fn_task1",  # ok
    suite=["community"],  # ok
    hf_repo="deutsche-telekom/Ger-RAG-eval",  # ok
    hf_subset="task1",  # ok
    hf_avail_splits=["test"],  # ok
    evaluation_splits=["test"],  # ok
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc"],  # ok
)

QUERY_TASK1: str = """\
Welche der folgenden Fragen (A oder B oder C oder D) lässt sich anhand des Kontext beantworten?

Kontext:
{context}

Fragen:
A: {choice_a}
B: {choice_b}
C: {choice_c}
D: {choice_d}
"""


def prompt_fn_task1(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """
    query = QUERY_TASK1.format(
        context=line["context"],
        choice_a=line["choice_a"],
        choice_b=line["choice_b"],
        choice_c=line["choice_c"],
        choice_d=line["choice_d"],
    )
    choices = ["A", "B", "C", "D"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


# STORE YOUR EVALS
_TASKS = [task1]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))

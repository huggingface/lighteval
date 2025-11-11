"""
name:
Openbookqa

dataset:
allenai/openbookqa

abstract:
OpenBookQA is a question-answering dataset modeled after open-book exams for
assessing human understanding of a subject. It contains multiple-choice
questions that require combining facts from a given open book with broad common
knowledge. The task tests language models' ability to leverage provided
information and apply common sense reasoning.

languages:
english

tags:
multiple-choice, qa

paper:
https://arxiv.org/abs/1809.02789
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


openbookqa = LightevalTaskConfig(
    name="openbookqa",
    prompt_function=prompt.openbookqa_helm,
    hf_repo="allenai/openbookqa",
    hf_subset="main",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    openbookqa,
]

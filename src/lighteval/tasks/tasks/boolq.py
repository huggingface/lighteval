"""
name:
Boolq

dataset:
lighteval/boolq_helm

abstract:
The BoolQ benchmark for binary (yes/no) question answering.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1905.11946
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


boolq = LightevalTaskConfig(
    name="boolq",
    prompt_function=prompt.boolq_helm,
    hf_repo="lighteval/boolq_helm",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


boolq_contrastset = LightevalTaskConfig(
    name="boolq:contrastset",
    prompt_function=prompt.boolq_helm_contrastset,
    hf_repo="lighteval/boolq_helm",
    hf_subset="default",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    boolq,
    boolq_contrastset,
]

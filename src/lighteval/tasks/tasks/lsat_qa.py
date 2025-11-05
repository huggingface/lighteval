"""
name:
Lsat Qa

dataset:
lighteval/lsat_qa

abstract:
Questions from law school admission tests.

languages:
english

tags:
legal, qa

paper:
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


lsat_qa = LightevalTaskConfig(
    name="lsat_qa",
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="all",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_assignment = LightevalTaskConfig(
    name="lsat_qa:assignment",
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="assignment",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_grouping = LightevalTaskConfig(
    name="lsat_qa:grouping",
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="grouping",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_miscellaneous = LightevalTaskConfig(
    name="lsat_qa:miscellaneous",
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="miscellaneous",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_ordering = LightevalTaskConfig(
    name="lsat_qa:ordering",
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="ordering",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    lsat_qa,
    lsat_qa_assignment,
    lsat_qa_grouping,
    lsat_qa_miscellaneous,
    lsat_qa_ordering,
]

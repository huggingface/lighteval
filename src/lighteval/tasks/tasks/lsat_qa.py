"""
abstract:
Questions from law school admission tests.

languages:
en

tags:
legal, qa

paper:
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


lsat_qa = LightevalTaskConfig(
    name="lsat_qa",
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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

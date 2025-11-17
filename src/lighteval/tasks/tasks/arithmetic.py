"""
name:
Arithmetic

dataset:
EleutherAI/arithmetic

abstract:
A small battery of 10 tests that involve asking language models a simple
arithmetic problem in natural language.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2005.14165
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


arithmetic_1dc = LightevalTaskConfig(
    name="arithmetic:1dc",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_1dc",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_2da = LightevalTaskConfig(
    name="arithmetic:2da",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_2da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_2dm = LightevalTaskConfig(
    name="arithmetic:2dm",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_2dm",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_2ds = LightevalTaskConfig(
    name="arithmetic:2ds",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_2ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_3da = LightevalTaskConfig(
    name="arithmetic:3da",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_3da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_3ds = LightevalTaskConfig(
    name="arithmetic:3ds",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_3ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_4da = LightevalTaskConfig(
    name="arithmetic:4da",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_4da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_4ds = LightevalTaskConfig(
    name="arithmetic:4ds",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_4ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_5da = LightevalTaskConfig(
    name="arithmetic:5da",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_5da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_5ds = LightevalTaskConfig(
    name="arithmetic:5ds",
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_5ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    arithmetic_1dc,
    arithmetic_2da,
    arithmetic_2dm,
    arithmetic_2ds,
    arithmetic_3da,
    arithmetic_3ds,
    arithmetic_4da,
    arithmetic_4ds,
    arithmetic_5da,
    arithmetic_5ds,
]

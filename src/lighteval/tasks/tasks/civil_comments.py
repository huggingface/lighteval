"""
name:
Civil Comments

dataset:
lighteval/civil_comments_helm

abstract:
The CivilComments benchmark for toxicity detection.

languages:
english

tags:
bias, classification

paper:
https://arxiv.org/abs/1903.04561
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


civil_comments = LightevalTaskConfig(
    name="civil_comments",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="all",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_LGBTQ = LightevalTaskConfig(
    name="civil_comments:LGBTQ",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="LGBTQ",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_black = LightevalTaskConfig(
    name="civil_comments:black",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="black",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_christian = LightevalTaskConfig(
    name="civil_comments:christian",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="christian",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_female = LightevalTaskConfig(
    name="civil_comments:female",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="female",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_male = LightevalTaskConfig(
    name="civil_comments:male",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="male",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_muslim = LightevalTaskConfig(
    name="civil_comments:muslim",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="muslim",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_other_religions = LightevalTaskConfig(
    name="civil_comments:other_religions",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="other_religions",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_white = LightevalTaskConfig(
    name="civil_comments:white",
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="white",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    civil_comments,
    civil_comments_LGBTQ,
    civil_comments_black,
    civil_comments_christian,
    civil_comments_female,
    civil_comments_male,
    civil_comments_muslim,
    civil_comments_other_religions,
    civil_comments_white,
]

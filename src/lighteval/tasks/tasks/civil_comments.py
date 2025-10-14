"""
abstract:
The CivilComments benchmark for toxicity detection.

languages:
en

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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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

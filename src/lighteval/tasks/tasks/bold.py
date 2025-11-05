"""
name:
Bold

dataset:
lighteval/bold_helm

abstract:
The Bias in Open-Ended Language Generation Dataset (BOLD) for measuring biases
and toxicity in open-ended language generation.

languages:
english

tags:
bias, generation

paper:
https://dl.acm.org/doi/10.1145/3442188.3445924
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


bold = LightevalTaskConfig(
    name="bold",
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="all",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_gender = LightevalTaskConfig(
    name="bold:gender",
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="gender",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_political_ideology = LightevalTaskConfig(
    name="bold:political_ideology",
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="political_ideology",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_profession = LightevalTaskConfig(
    name="bold:profession",
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="profession",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_race = LightevalTaskConfig(
    name="bold:race",
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="race",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_religious_ideology = LightevalTaskConfig(
    name="bold:religious_ideology",
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="religious_ideology",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    bold,
    bold_gender,
    bold_political_ideology,
    bold_profession,
    bold_race,
    bold_religious_ideology,
]

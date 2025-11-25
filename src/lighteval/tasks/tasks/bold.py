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

from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def bold_prompt(line, task_name: str = None):
    return Doc(task_name=task_name, query=line["text"], choices=None, gold_index=None)


def record_to_sample(record):
    query = record["text"]
    target = ""
    return Sample(input=query, target=target)


bold = LightevalTaskConfig(
    name="bold",
    prompt_function=bold_prompt,
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
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
)

bold_gender = LightevalTaskConfig(
    name="bold:gender",
    prompt_function=bold_prompt,
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
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
)

bold_political_ideology = LightevalTaskConfig(
    name="bold:political_ideology",
    prompt_function=bold_prompt,
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
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
)

bold_profession = LightevalTaskConfig(
    name="bold:profession",
    prompt_function=bold_prompt,
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
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
)

bold_race = LightevalTaskConfig(
    name="bold:race",
    prompt_function=bold_prompt,
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
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
)

bold_religious_ideology = LightevalTaskConfig(
    name="bold:religious_ideology",
    prompt_function=bold_prompt,
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
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
)

TASKS_TABLE = [
    bold,
    bold_gender,
    bold_political_ideology,
    bold_profession,
    bold_race,
    bold_religious_ideology,
]

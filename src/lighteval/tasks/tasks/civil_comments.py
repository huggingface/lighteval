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

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def civil_comments_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['text']}\nIs the passage above toxic?\nAnswer: ",
        choices=[str(line["gold"])],
        gold_index=0,
    )


def record_to_sample(record):
    query = f"{record['text']}\nIs the passage above toxic?"
    choices = ["True", "False"]
    target = ascii_uppercase[choices.index(record["gold"])]
    return Sample(input=query, target=target, choices=choices)


civil_comments_LGBTQ = LightevalTaskConfig(
    name="civil_comments:LGBTQ",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

civil_comments_black = LightevalTaskConfig(
    name="civil_comments:black",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

civil_comments_christian = LightevalTaskConfig(
    name="civil_comments:christian",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

civil_comments_female = LightevalTaskConfig(
    name="civil_comments:female",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

civil_comments_male = LightevalTaskConfig(
    name="civil_comments:male",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

civil_comments_muslim = LightevalTaskConfig(
    name="civil_comments:muslim",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

civil_comments_other_religions = LightevalTaskConfig(
    name="civil_comments:other_religions",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

civil_comments_white = LightevalTaskConfig(
    name="civil_comments:white",
    prompt_function=civil_comments_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    civil_comments_LGBTQ,
    civil_comments_black,
    civil_comments_christian,
    civil_comments_female,
    civil_comments_male,
    civil_comments_muslim,
    civil_comments_other_religions,
    civil_comments_white,
]

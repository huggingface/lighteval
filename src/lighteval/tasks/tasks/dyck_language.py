"""
name:
Dyck Language

dataset:
lighteval/DyckLanguage

abstract:
Scenario testing hierarchical reasoning through the Dyck formal languages.

languages:
english

tags:
reasoning

paper:
https://aclanthology.org/W19-3905/
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


PROMPT = "Please complete the rest of the following Dyck sequences, making sure that the parentheses are closed properly.\n Input: {prompt}"


def record_to_sample(record):
    return Sample(input=PROMPT.format(prompt=record["input"]), target=record["output"])


def dyck_language_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Please complete the rest of the following Dyck sequences, making sure that the parentheses are closed properly.\n Input: {line['input']}",
        choices=[line["output"]],
        gold_index=0,
        instruction="Please complete the rest of the following Dyck sequences, making sure that the parentheses are closed properly.\n ",
    )


dyck_language_2 = LightevalTaskConfig(
    name="dyck_language:2",
    prompt_function=dyck_language_prompt,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="2",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


dyck_language_3 = LightevalTaskConfig(
    name="dyck_language:3",
    prompt_function=dyck_language_prompt,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="3",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


dyck_language_4 = LightevalTaskConfig(
    name="dyck_language:4",
    prompt_function=dyck_language_prompt,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="4",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    dyck_language_2,
    dyck_language_3,
    dyck_language_4,
]

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

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def boolq_prompt(line, task_name: str = None):
    question = line["question"][:-1] if line["question"][-2:] == "??" else line["question"]
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {question}\nAnswer:",
        choices=[" Yes", " No"],
        gold_index=["Yes", "No"].index(line["answer"]),
    )


def boolq_contrastset_prompt(line, task_name: str = None):
    if line["contrast_inputs"] in [None, ""]:
        return boolq_prompt(line)

    return [
        Doc(
            task_name=task_name,
            query=f"{passage}\nQuestion: {question}\nAnswer:",
            choices=["Yes", "No"],
            gold_index=["No", "Yes"].index(line["answer"]),
        )
        for passage, question in zip(line["contrast_inputs"]["passage"], line["contrast_inputs"]["question"])
    ][0]


def record_to_sample(record):
    choices = ["Yes", "No"]
    query = f"{record['passage']}\n{record['question']}"
    target = ascii_uppercase[choices.index(record["answer"])]
    return Sample(input=query, target=target, choices=choices)


def record_to_sample_contrastset(record):
    if record["contrast_inputs"] in [None, ""]:
        return record_to_sample(record)

    choices = ["Yes", "No"]
    query = f"{record['contrast_inputs']['passage']}\n{record['contrast_inputs']['question']}"
    target = ascii_uppercase[choices.index(record["answer"])]

    return Sample(input=query, target=target, choices=choices)


boolq = LightevalTaskConfig(
    name="boolq",
    prompt_function=boolq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)


boolq_contrastset = LightevalTaskConfig(
    name="boolq:contrastset",
    prompt_function=boolq_contrastset_prompt,
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
    sample_fields=record_to_sample_contrastset,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    boolq,
    boolq_contrastset,
]

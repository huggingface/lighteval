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
)

TASKS_TABLE = [
    boolq,
    boolq_contrastset,
]

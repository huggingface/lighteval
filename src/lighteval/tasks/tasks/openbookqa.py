"""
name:
Openbookqa

dataset:
allenai/openbookqa

abstract:
OpenBookQA is a question-answering dataset modeled after open-book exams for
assessing human understanding of a subject. It contains multiple-choice
questions that require combining facts from a given open book with broad common
knowledge. The task tests language models' ability to leverage provided
information and apply common sense reasoning.

languages:
english

tags:
multiple-choice, qa

paper:
https://arxiv.org/abs/1809.02789
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def openbookqa_prompt(line, task_name: str = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['question_stem']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["choices"]["text"])])
    query += "Answer: "

    gold_ix = ["A", "B", "C", "D", "E"].index(line["answerKey"].strip())
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase[: len(line["choices"]["text"])]),
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


openbookqa = LightevalTaskConfig(
    name="openbookqa",
    prompt_function=openbookqa_prompt,
    hf_repo="allenai/openbookqa",
    hf_subset="main",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    openbookqa,
]

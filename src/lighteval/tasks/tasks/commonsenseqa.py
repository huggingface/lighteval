"""
name:
Commonsenseqa

dataset:
tau/commonsense_qa

abstract:
CommonsenseQA is a new multiple-choice question answering dataset that requires
different types of commonsense knowledge to predict the correct answers . It
contains 12,102 questions with one correct answer and four distractor answers.
The dataset is provided in two major training/validation/testing set splits:
"Random split" which is the main evaluation split, and "Question token split",
see paper for details.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
https://arxiv.org/abs/1811.00937
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def commonsenseqa_prompt(line, task_name: str = None):
    query = f"The following are multiple choice questions (with answers) about common sense.\nQuestion: {line['question']}\n"
    query += "".join(
        [f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, [f" {c}" for c in line["choices"]["text"]])]
    )
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[: len(line["choices"]["text"])],
        gold_index=list(ascii_uppercase).index(line["answerKey"].strip()),
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


def record_to_sample(record):
    query = record["question"]
    choices = record["choices"]["text"]
    target = record["answerKey"]
    return Sample(input=query, target=target, choices=choices)


commonsenseqa = LightevalTaskConfig(
    name="commonsenseqa",
    prompt_function=commonsenseqa_prompt,
    hf_repo="tau/commonsense_qa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    commonsenseqa,
]

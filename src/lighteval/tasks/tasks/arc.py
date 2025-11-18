"""
name:
Arc

dataset:
allenai/ai2_arc

abstract:
7,787 genuine grade-school level, multiple-choice science questions, assembled
to encourage research in advanced question-answering. The dataset is partitioned
into a Challenge Set and an Easy Set, where the former contains only questions
answered incorrectly by both a retrieval-based algorithm and a word
co-occurrence algorithm

languages:
english

tags:
multiple-choice

paper:
https://arxiv.org/abs/1803.05457
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def arc_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


arc_challenge = LightevalTaskConfig(
    name="arc:challenge",
    prompt_function=arc_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

arc_easy = LightevalTaskConfig(
    name="arc:easy",
    prompt_function=arc_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [arc_challenge, arc_easy]

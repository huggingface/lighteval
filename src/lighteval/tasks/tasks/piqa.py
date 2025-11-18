"""
name:
Piqa

dataset:
ybisk/piqa

abstract:
PIQA is a benchmark for testing physical commonsense reasoning. It contains
questions requiring this kind of physical commonsense reasoning.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
https://arxiv.org/abs/1911.11641
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def piqa_prompt(line, task_name: str = None):
    letters = list(ascii_uppercase)[:2]
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['goal']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(letters, [line["sol1"], line["sol2"]])])
    query += "Answer: "

    gold_ix = int(line["label"])
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=query,
        choices=letters if not is_few_shots else [line["sol1"], line["sol2"]],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


piqa = LightevalTaskConfig(
    name="piqa",
    prompt_function=piqa_prompt,
    hf_repo="ybisk/piqa",
    hf_subset="plain_text",
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
    piqa,
]

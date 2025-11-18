"""
name:
Logiqa

dataset:
lighteval/logiqa_harness

abstract:
LogiQA is a machine reading comprehension dataset focused on testing logical
reasoning abilities. It contains 8,678 expert-written multiple-choice questions
covering various types of deductive reasoning. While humans perform strongly,
state-of-the-art models lag far behind, making LogiQA a benchmark for advancing
logical reasoning in NLP systems.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/2007.08124
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def logiqa_prompt(line, task_name: str = None):
    query = f"Passage: {line['context']}\nQuestion: {line['question']}\nChoices:\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(["A", "B", "C", "D"], line["options"])])
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {c}" for c in line["options"]],
        gold_index=["a", "b", "c", "d"].index(line["label"]),
    )


logiqa = LightevalTaskConfig(
    name="logiqa",
    prompt_function=logiqa_prompt,
    hf_repo="lighteval/logiqa_harness",
    hf_subset="logiqa",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    logiqa,
]

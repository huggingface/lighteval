"""
name:
Headqa

dataset:
lighteval/headqa_harness

abstract:
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to
access a specialized position in the Spanish healthcare system, and are
challenging even for highly specialized humans. They are designed by the
Ministerio de Sanidad, Consumo y Bienestar Social, who also provides direct
access to the exams of the last 5 years.

languages:
english, spanish

tags:
health, medical, multiple-choice, qa

paper:
https://arxiv.org/abs/1906.04701
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def headqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['qtext']}\nAnswer:",
        choices=[f" {answer['atext']}" for answer in line["answers"]],
        gold_index=int(line["ra"]) - 1,
    )


headqa_en = LightevalTaskConfig(
    name="headqa:en",
    prompt_function=headqa_prompt,
    hf_repo="lighteval/headqa_harness",
    hf_subset="en",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)


headqa_es = LightevalTaskConfig(
    name="headqa:es",
    prompt_function=headqa_prompt,
    hf_repo="lighteval/headqa_harness",
    hf_subset="es",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    headqa_en,
    headqa_es,
]

"""
name:
Prost

dataset:
lighteval/prost

abstract:
PROST is a benchmark for testing physical reasoning about objects through space
and time. It includes 18,736 multiple-choice questions covering 10 core physics
concepts, designed to probe models in zero-shot settings. Results show that even
large pretrained models struggle with physical reasoning and are sensitive to
question phrasing, underscoring their limited real-world understanding.

languages:
english

tags:
reasoning, qa, physical-commonsense

paper:
https://arxiv.org/abs/2106.03634
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prost_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]],
        gold_index=int(line["label"]) if isinstance(line["label"], int) else int(line["label"]),
    )


prost = LightevalTaskConfig(
    name="prost",
    prompt_function=prost_prompt,
    hf_repo="lighteval/prost",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    prost,
]

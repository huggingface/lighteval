"""
name:
Quac

dataset:
lighteval/quac_helm

abstract:
The QuAC benchmark for question answering in the context of dialogues.

languages:
english

tags:
dialog, qa

paper:
https://aclanthology.org/D18-1241/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def quac_prompt(line, task_name: str = None):
    references = [ref for ref in line["references"] if ref is not None and ref != ""]
    return Doc(
        task_name=task_name,
        query=f"{line['prompt']}\nAnswer:",
        choices=references,
        gold_index=list(range(len(references))),
    )


quac = LightevalTaskConfig(
    name="quac",
    prompt_function=quac_prompt,
    hf_repo="lighteval/quac_helm",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    quac,
]

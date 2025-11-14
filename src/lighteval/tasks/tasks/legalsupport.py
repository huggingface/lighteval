"""
name:
Legalsupport

dataset:
lighteval/LegalSupport

abstract:
Measures fine-grained legal reasoning through reverse entailment.

languages:
english

tags:
legal

paper:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def legalsupport_prompt(line, task_name: str = None):
    query = f"Which statement best supports the passage?\nPassage: {line['context']}\n"
    query += "".join(
        [
            f"{key}. {choice}\n"
            for key, choice in zip(
                ["a", "b"], [line["citation_a"]["parenthetical"], line["citation_b"]["parenthetical"]]
            )
        ]
    )
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=["a", "b"],
        gold_index=0 if line["answer_label"] == "citation_a" else 1,
    )


legalsupport = LightevalTaskConfig(
    name="legalsupport",
    prompt_function=legalsupport_prompt,
    hf_repo="lighteval/LegalSupport",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    legalsupport,
]

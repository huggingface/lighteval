"""
name:
Pubmedqa

dataset:
pubmed_qa

abstract:
PubMedQA is a dataset for biomedical research question answering.

languages:
english

tags:
biomedical, health, medical, qa

paper:
https://pubmedqa.github.io/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def pubmed_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['QUESTION']}\n{line['CONTEXTS']}\nAnswer: ",
        choices=[line["final_decision"]],
        gold_index=0,
    )


pubmedqa = LightevalTaskConfig(
    name="pubmedqa",
    prompt_function=pubmed_qa_prompt,
    hf_repo="pubmed_qa",
    hf_subset="pqa_labeled",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
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
    pubmedqa,
]

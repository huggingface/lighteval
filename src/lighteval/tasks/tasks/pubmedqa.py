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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


pubmedqa = LightevalTaskConfig(
    name="pubmedqa",
    prompt_function=prompt.pubmed_qa_helm,
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

"""
name:
Med

dataset:
lighteval/med_mcqa, lighteval/med_paragraph_simplification, bigbio/med_qa

abstract:
A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering

languages:
english

tags:
health, medical

paper:
https://medmcqa.github.io/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


med_mcqa = LightevalTaskConfig(
    name="med_mcqa",
    prompt_function=prompt.med_mcqa,
    hf_repo="lighteval/med_mcqa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


med_paragraph_simplification = LightevalTaskConfig(
    name="med_paragraph_simplification",
    prompt_function=prompt.med_paragraph_simplification,
    hf_repo="lighteval/med_paragraph_simplification",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


med_qa = LightevalTaskConfig(
    name="med_qa",
    prompt_function=prompt.med_qa,
    hf_repo="bigbio/med_qa",
    hf_subset="med_qa_en_source",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    med_mcqa,
    med_paragraph_simplification,
    med_qa,
]

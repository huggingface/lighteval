"""
name:
Med Dialog

dataset:
lighteval/med_dialog

abstract:
A collection of medical dialogue datasets.

languages:
english

tags:
dialog, health, medical

paper:
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


med_dialog_healthcaremagic = LightevalTaskConfig(
    name="med_dialog:healthcaremagic",
    prompt_function=prompt.med_dialog,
    hf_repo="lighteval/med_dialog",
    hf_subset="healthcaremagic",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=128,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


med_dialog_icliniq = LightevalTaskConfig(
    name="med_dialog:icliniq",
    prompt_function=prompt.med_dialog,
    hf_repo="lighteval/med_dialog",
    hf_subset="icliniq",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=128,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    med_dialog_healthcaremagic,
    med_dialog_icliniq,
]

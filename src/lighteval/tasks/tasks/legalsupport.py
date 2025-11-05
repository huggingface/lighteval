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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


legalsupport = LightevalTaskConfig(
    name="legalsupport",
    prompt_function=prompt.legal_support,
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

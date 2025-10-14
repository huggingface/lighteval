"""
abstract:
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to
access a specialized position in the Spanish healthcare system, and are
challenging even for highly specialized humans. They are designed by the
Ministerio de Sanidad, Consumo y Bienestar Social, who also provides direct
access to the exams of the last 5 years.

languages:
en, es

tags:
health, reasoning

paper:
https://arxiv.org/abs/1906.04701
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


headqa_en = LightevalTaskConfig(
    name="headqa:en",
    suite=["lighteval"],
    prompt_function=prompt.headqa,
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
    suite=["lighteval"],
    prompt_function=prompt.headqa,
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

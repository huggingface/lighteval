"""
name:
Gsm Plus

dataset:
qintongli/GSM-Plus

abstract:
GSM-Plus is an adversarial extension of GSM8K that tests the robustness of LLMs'
mathematical reasoning by introducing varied perturbations to grade-school math
problems.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2402.19255
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


gsm_plus = LightevalTaskConfig(
    name="gsm_plus",
    suite=["lighteval"],
    prompt_function=prompt.gsm_plus,
    hf_repo="qintongli/GSM-Plus",
    hf_subset="default",
    hf_avail_splits=["test", "testmini"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=None,
    version=0,
)

TASKS_TABLE = [
    gsm_plus,
]

"""
abstract:
PIQA is a benchmark for testing physical commonsense reasoning. It contains
questions requiring this kind of physical commonsense reasoning.

languages:
en

tags:
reasoning, physical-commonsense, qa

paper:
https://arxiv.org/abs/1911.11641
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


piqa = LightevalTaskConfig(
    name="piqa",
    suite=["lighteval"],
    prompt_function=prompt.piqa_helm,
    hf_repo="ybisk/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

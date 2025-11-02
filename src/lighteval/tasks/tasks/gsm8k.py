"""
name:
Gsm8K

dataset:
openai/gsm8k

abstract:
GSM8K is a dataset of 8,000+ high-quality, single-step arithmetic word problems.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2110.14168
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


gsm8k = LightevalTaskConfig(
    name="gsm8k",
    suite=["lighteval"],
    prompt_function=prompt.gsm8k,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metrics=[
        Metrics.expr_gold_metric,
    ],
    stop_sequence=["Question:"],
    version=0,
)

TASKS_TABLE = [
    gsm8k,
]

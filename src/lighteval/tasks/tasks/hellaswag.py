"""
name:
Hellaswag

dataset:
Rowan/hellaswag

abstract:
HellaSwag is a commonsense inference benchmark designed to challenge language
models with adversarially filtered multiple-choice questions.

languages:
english

tags:
multiple-choice, narrative, reasoning

paper:
https://arxiv.org/abs/1905.07830
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


hellaswag = LightevalTaskConfig(
    name="hellaswag",
    prompt_function=prompt.hellaswag_generative,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
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
    hellaswag,
]

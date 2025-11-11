"""
name:
Toxigen

dataset:
skg/toxigen-data

abstract:
This dataset is for implicit hate speech detection. All instances were generated
using GPT-3 and the methods described in our paper.

languages:
english

tags:
generation, safety

paper:
https://arxiv.org/abs/2203.09509
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


toxigen = LightevalTaskConfig(
    name="toxigen",
    prompt_function=prompt.toxigen,
    hf_repo="skg/toxigen-data",
    hf_subset="annotated",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    toxigen,
]

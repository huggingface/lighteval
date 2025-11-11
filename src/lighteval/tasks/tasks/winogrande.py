"""
name:
Winogrande

dataset:
allenai/winogrande

abstract:
WinoGrande is a new collection of 44k problems, inspired by Winograd Schema
Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the
scale and robustness against the dataset-specific bias. Formulated as a
fill-in-a-blank task with binary options, the goal is to choose the right option
for a given sentence which requires commonsense reasoning.

languages:
english

tags:
commonsense, multiple-choice

paper:
https://arxiv.org/abs/1907.10641
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


winogrande = LightevalTaskConfig(
    name="winogrande",
    prompt_function=prompt.winogrande,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    winogrande,
]

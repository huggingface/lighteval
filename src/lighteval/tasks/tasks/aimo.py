"""
name:
AIMO Progress Prize 1

dataset:
https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize

abstract:
Task to evaluate LLMs on the training set of the Kaggle AIMO competition:

languages:
english

tags:
math, reasoning

paper:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import math_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def aimo_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        choices=[str(line["answer"])],
        gold_index=0,
        query=line["problem"],
    )


task = LightevalTaskConfig(
    name="aimo_progress_prize_1",
    prompt_function=aimo_prompt,
    hf_subset="",
    hf_repo="lighteval/aimo_progress_prize_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="sequential",
    metrics=[
        Metrics.exact_match(sample_params={"normalize_gold": math_normalizer, "normalize_pred": math_normalizer})
    ],
    generation_size=2048,
    stop_sequence=None,
)

# STORE YOUR EVALS
TASKS_TABLE = [task]

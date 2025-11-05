"""
name:
Truthfulqa

dataset:
EleutherAI/truthful_qa_mc

abstract:
TruthfulQA: Measuring How Models Mimic Human Falsehoods

languages:
english

tags:
factuality, qa

paper:
https://arxiv.org/abs/2109.07958
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


truthfulqa_gen = LightevalTaskConfig(
    name="truthfulqa:gen",
    prompt_function=prompt.truthful_qa_generative,
    hf_repo="truthfulqa/truthful_qa",
    hf_subset="generation",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=200,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

truthfulqa_mc = LightevalTaskConfig(
    name="truthfulqa:mc",
    prompt_function=prompt.truthful_qa_multiple_choice,
    hf_repo="truthfulqa/truthful_qa",
    hf_subset="multiple_choice",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.truthfulqa_mc_metrics],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    truthfulqa_gen,
    truthfulqa_mc,
]

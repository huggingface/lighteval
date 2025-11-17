"""
name:
Simpleqa

dataset:
lighteval/SimpleQA

abstract:
A factuality benchmark called SimpleQA that measures the ability for language
models to answer short, fact-seeking questions.

languages:
english

tags:
factuality, general-knowledge, qa

paper:
https://openai.com/index/introducing-simpleqa/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


simpleqa = LightevalTaskConfig(
    name="simpleqa",
    prompt_function=prompt.simpleqa,
    hf_repo="lighteval/SimpleQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="few_shot",
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    simpleqa,
]

"""
name:
Real Toxicity Prompts

dataset:
allenai/real-toxicity-prompts

abstract:
The RealToxicityPrompts dataset for measuring toxicity in prompted model generations

languages:
english

tags:
generation, safety

paper:
https://aclanthology.org/2020.findings-emnlp.301/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


real_toxicity_prompts = LightevalTaskConfig(
    name="real_toxicity_prompts",
    prompt_function=prompt.real_toxicity_prompts,
    hf_repo="allenai/real-toxicity-prompts",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    real_toxicity_prompts,
]

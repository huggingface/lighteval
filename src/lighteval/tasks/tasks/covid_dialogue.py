"""
name:
Covid Dialogue

dataset:
lighteval/covid_dialogue

abstract:
The COVID-19 Dialogue dataset is a collection of 500+ dialogues between
doctors and patients during the COVID-19 pandemic.

languages:
english

tags:
dialog, medical

paper:
https://arxiv.org/abs/2004.06561
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


covid_dialogue = LightevalTaskConfig(
    name="covid_dialogue",
    prompt_function=prompt.covid_dialogue,
    hf_repo="lighteval/covid_dialogue",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=128,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    covid_dialogue,
]

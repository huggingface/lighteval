"""
abstract:
A Corpus and Cloze Evaluation for Deeper Understanding of
Commonsense Stories

languages:
en

tags:
narrative, reasoning

paper:
https://arxiv.org/abs/1604.01696
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


storycloze_2016 = LightevalTaskConfig(
    name="storycloze:2016",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="MoE-UNC/story_cloze",
    hf_subset="2016",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


storycloze_2018 = LightevalTaskConfig(
    name="storycloze:2018",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="MoE-UNC/story_cloze",
    hf_subset="2018",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

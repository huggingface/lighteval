"""
name:
Xwinograd

dataset:
Muennighoff/xwinograd

abstract:
Multilingual winograd schema challenge as used in Crosslingual Generalization through Multitask Finetuning.

languages:
english, french, japanese, portuguese, russian, chinese

tags:
commonsense, multilingual, reasoning

paper:
https://arxiv.org/abs/2211.01786
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


xwinograd_en = LightevalTaskConfig(
    name="xwinograd:en",
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_fr = LightevalTaskConfig(
    name="xwinograd:fr",
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_jp = LightevalTaskConfig(
    name="xwinograd:jp",
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="jp",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_pt = LightevalTaskConfig(
    name="xwinograd:pt",
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="pt",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_ru = LightevalTaskConfig(
    name="xwinograd:ru",
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_zh = LightevalTaskConfig(
    name="xwinograd:zh",
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="zh",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    xwinograd_en,
    xwinograd_fr,
    xwinograd_jp,
    xwinograd_pt,
    xwinograd_ru,
    xwinograd_zh,
]

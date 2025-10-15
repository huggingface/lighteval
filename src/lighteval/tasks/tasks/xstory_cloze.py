"""
name:
Xstory Cloze

dataset:
juletxara/xstory_cloze

abstract:
XStoryCloze consists of the professionally translated version of the English
StoryCloze dataset (Spring 2016 version) to 10 non-English languages. This
dataset is released by Meta AI.

languages:
english, russian, chinese, spanish, arabic, hindi, indonesian, telugu, swahili, basque, burmese

tags:
multilingual, narrative, reasoning

paper:
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


xstory_cloze_en = LightevalTaskConfig(
    name="xstory_cloze:en",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="en",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_ru = LightevalTaskConfig(
    name="xstory_cloze:ru",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="ru",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_zh = LightevalTaskConfig(
    name="xstory_cloze:zh",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="zh",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_es = LightevalTaskConfig(
    name="xstory_cloze:es",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="es",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_ar = LightevalTaskConfig(
    name="xstory_cloze:ar",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="ar",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_hi = LightevalTaskConfig(
    name="xstory_cloze:hi",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="hi",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_id = LightevalTaskConfig(
    name="xstory_cloze:id",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="id",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_te = LightevalTaskConfig(
    name="xstory_cloze:te",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="te",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_sw = LightevalTaskConfig(
    name="xstory_cloze:sw",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="sw",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_eu = LightevalTaskConfig(
    name="xstory_cloze:eu",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="eu",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_my = LightevalTaskConfig(
    name="xstory_cloze:my",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="my",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

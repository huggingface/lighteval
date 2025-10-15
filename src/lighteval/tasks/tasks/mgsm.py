"""
name:
Mgsm

dataset:
juletxara/mgsm

abstract:
Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school
math problems.
The same 250 problems from GSM8K are each translated via human annotators in 10
languages.

languages:
english, spanish, french, german, russian, chinese, japanese, thai, swahili, bengali, telugu

tags:
math, multilingual, reasoning

paper:
https://arxiv.org/abs/2210.03057
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


mgsm_en = LightevalTaskConfig(
    name="mgsm:en",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_en,
    hf_repo="juletxara/mgsm",
    hf_subset="en",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_es = LightevalTaskConfig(
    name="mgsm:es",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_es,
    hf_repo="juletxara/mgsm",
    hf_subset="es",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_fr = LightevalTaskConfig(
    name="mgsm:fr",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_fr,
    hf_repo="juletxara/mgsm",
    hf_subset="fr",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_de = LightevalTaskConfig(
    name="mgsm:de",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_de,
    hf_repo="juletxara/mgsm",
    hf_subset="de",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_ru = LightevalTaskConfig(
    name="mgsm:ru",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_ru,
    hf_repo="juletxara/mgsm",
    hf_subset="ru",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_zh = LightevalTaskConfig(
    name="mgsm:zh",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_zh,
    hf_repo="juletxara/mgsm",
    hf_subset="zh",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_ja = LightevalTaskConfig(
    name="mgsm:ja",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_ja,
    hf_repo="juletxara/mgsm",
    hf_subset="ja",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_th = LightevalTaskConfig(
    name="mgsm:th",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_th,
    hf_repo="juletxara/mgsm",
    hf_subset="th",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_sw = LightevalTaskConfig(
    name="mgsm:sw",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_sw,
    hf_repo="juletxara/mgsm",
    hf_subset="sw",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_bn = LightevalTaskConfig(
    name="mgsm:bn",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_bn,
    hf_repo="juletxara/mgsm",
    hf_subset="bn",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

mgsm_te = LightevalTaskConfig(
    name="mgsm:te",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_te,
    hf_repo="juletxara/mgsm",
    hf_subset="te",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

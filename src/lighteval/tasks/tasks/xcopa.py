"""
name:
Xcopa

dataset:
cambridgeltl/xcopa

abstract:
XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning The Cross-lingual
Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability
of machine learning models to transfer commonsense reasoning across languages.

languages:
english

tags:
commonsense, multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2005.00333
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


xcopa_en = LightevalTaskConfig(
    name="xcopa:en",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_en,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="default",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_et = LightevalTaskConfig(
    name="xcopa:et",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_et,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="et",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_ht = LightevalTaskConfig(
    name="xcopa:ht",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_ht,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="ht",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_it = LightevalTaskConfig(
    name="xcopa:it",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_it,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="it",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_id = LightevalTaskConfig(
    name="xcopa:id",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_id,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="id",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_qu = LightevalTaskConfig(
    name="xcopa:qu",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_qu,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="qu",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_sw = LightevalTaskConfig(
    name="xcopa:sw",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_sw,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="sw",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_zh = LightevalTaskConfig(
    name="xcopa:zh",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_zh,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="zh",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_ta = LightevalTaskConfig(
    name="xcopa:ta",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_ta,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="ta",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_th = LightevalTaskConfig(
    name="xcopa:th",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_th,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="th",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_tr = LightevalTaskConfig(
    name="xcopa:tr",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_tr,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="tr",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_vi = LightevalTaskConfig(
    name="xcopa:vi",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_vi,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="vi",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

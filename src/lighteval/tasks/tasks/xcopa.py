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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def xcopa(line, connectors: dict, task_name: str = None):
    text = line["premise"]
    question = line["question"]
    connector = connectors[question]
    query = f"Premise: {text}\nQuestion: {connector}"
    choices = [f" {line['choice1']}", f" {line['choice2']}"]
    gold_index = int(line["label"]) - 1 if isinstance(line["label"], str) else int(line["label"])
    return Doc(task_name=task_name, query=query, choices=choices, gold_index=gold_index)


def xcopa_en(line, task_name: str = None):
    return xcopa(line, {"cause": "because", "effect": "therefore"}, task_name)


def xcopa_et(line, task_name: str = None):
    return xcopa(line, {"cause": "sest", "effect": "seet\u00f6ttu"}, task_name)


def xcopa_ht(line, task_name: str = None):
    return xcopa(line, {"cause": "paske", "effect": "donc"}, task_name)


def xcopa_it(line, task_name: str = None):
    return xcopa(line, {"cause": "perch\u00e9", "effect": "quindi"}, task_name)


def xcopa_id(line, task_name: str = None):
    return xcopa(line, {"cause": "karena", "effect": "oleh karena itu"}, task_name)


def xcopa_qu(line, task_name: str = None):
    return xcopa(line, {"cause": "imarayku", "effect": "chayna\u00b4r\u00f0m"}, task_name)


def xcopa_sw(line, task_name: str = None):
    return xcopa(line, {"cause": "kwa sababu", "effect": "hivyo"}, task_name)


def xcopa_zh(line, task_name: str = None):
    return xcopa(line, {"cause": "因為", "effect": "因此"}, task_name)


def xcopa_ta(line, task_name: str = None):
    return xcopa(line, {"cause": "ஏனெனில்", "effect": "ஆகையால்"}, task_name)


def xcopa_th(line, task_name: str = None):
    return xcopa(line, {"cause": "เพราะ", "effect": "ดังนั้น"}, task_name)


def xcopa_tr(line, task_name: str = None):
    return xcopa(line, {"cause": "\u00e7\u00fc\u0308nk\u00fc", "effect": "bu y\u00fczden"}, task_name)


def xcopa_vi(line, task_name: str = None):
    return xcopa(line, {"cause": "b\u1edfi v\u00ec", "effect": "v\u00ec v\u1eady"}, task_name)


xcopa_en = LightevalTaskConfig(
    name="xcopa:en",
    prompt_function=xcopa_en,
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
    prompt_function=xcopa_et,
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
    prompt_function=xcopa_ht,
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
    prompt_function=xcopa_it,
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
    prompt_function=xcopa_id,
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
    prompt_function=xcopa_qu,
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
    prompt_function=xcopa_sw,
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
    prompt_function=xcopa_zh,
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
    prompt_function=xcopa_ta,
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
    prompt_function=xcopa_th,
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
    prompt_function=xcopa_tr,
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
    prompt_function=xcopa_vi,
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

TASKS_TABLE = [
    xcopa_en,
    xcopa_et,
    xcopa_ht,
    xcopa_it,
    xcopa_id,
    xcopa_qu,
    xcopa_sw,
    xcopa_zh,
    xcopa_ta,
    xcopa_th,
    xcopa_tr,
    xcopa_vi,
]

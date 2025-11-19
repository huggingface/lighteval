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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mgsm_prompt(line, question_key, answer_key, task_name: str = None):
    if line["answer"] is not None:
        query = f"{line['question']}\n{answer_key}"
        gold = f" {line['answer'][len(answer_key) + 1 :]}"
    else:
        query = f"{question_key} {line['question']}\n{answer_key}"
        gold = f" {str(line['answer_number'])}"
    return Doc(task_name=task_name, query=query, choices=[gold], gold_index=0)


def mgsm_en_prompt(line, task_name: str = None):
    question_key = "Question:"
    answer_key = "Step-by-Step Answer:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_es_prompt(line, task_name: str = None):
    question_key = "Pregunta:"
    answer_key = "Respuesta paso a paso:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_fr_prompt(line, task_name: str = None):
    question_key = "Question:"
    answer_key = "Réponse étape par étape :"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_de_prompt(line, task_name: str = None):
    question_key = "Frage:"
    answer_key = "Schritt-für-Schritt-Antwort:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_ru_prompt(line, task_name: str = None):
    question_key = "Задача:"
    answer_key = "Пошаговоерешение:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_zh_prompt(line, task_name: str = None):
    question_key = "问题:"
    answer_key = "逐步解答:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_ja_prompt(line, task_name: str = None):
    question_key = "問題:"
    answer_key = "ステップごとの答え:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_th_prompt(line, task_name: str = None):
    question_key = "โจทย์:"
    answer_key = "คำตอบทีละขั้นตอน:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_sw_prompt(line, task_name: str = None):
    question_key = "Swali:"
    answer_key = "Jibu la Hatua kwa Hatua:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_bn_prompt(line, task_name: str = None):
    question_key = "প্রশ্ন:"
    answer_key = "ধাপে ধাপে উত্তর:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_te_prompt(line, task_name: str = None):
    question_key = "ప్రశ్న:"
    answer_key = "దశలవారీగా సమాధానం:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


mgsm_en = LightevalTaskConfig(
    name="mgsm:en",
    prompt_function=mgsm_en_prompt,
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
    prompt_function=mgsm_es_prompt,
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
    prompt_function=mgsm_fr_prompt,
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
    prompt_function=mgsm_de_prompt,
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
    prompt_function=mgsm_ru_prompt,
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
    prompt_function=mgsm_zh_prompt,
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
    prompt_function=mgsm_ja_prompt,
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
    prompt_function=mgsm_th_prompt,
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
    prompt_function=mgsm_sw_prompt,
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
    prompt_function=mgsm_bn_prompt,
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
    prompt_function=mgsm_te_prompt,
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

TASKS_TABLE = [
    mgsm_en,
    mgsm_es,
    mgsm_fr,
    mgsm_de,
    mgsm_ru,
    mgsm_zh,
    mgsm_ja,
    mgsm_th,
    mgsm_sw,
    mgsm_bn,
    mgsm_te,
]

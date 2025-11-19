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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def storycloze_prompt(line, task_name: str = None):
    context = "\n".join(
        [line["input_sentence_1"], line["input_sentence_2"], line["input_sentence_3"], line["input_sentence_4"]]
    )
    choices = [line["sentence_quiz1"], line["sentence_quiz2"]]
    gold = int(line["answer_right_ending"]) - 1
    return Doc(task_name=task_name, query=context + "\n", choices=choices, gold_index=gold)


xstory_cloze_en = LightevalTaskConfig(
    name="xstory_cloze:en",
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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
    prompt_function=storycloze_prompt,
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

TASKS_TABLE = [
    xstory_cloze_en,
    xstory_cloze_ru,
    xstory_cloze_zh,
    xstory_cloze_es,
    xstory_cloze_ar,
    xstory_cloze_hi,
    xstory_cloze_id,
    xstory_cloze_te,
    xstory_cloze_sw,
    xstory_cloze_eu,
    xstory_cloze_my,
]

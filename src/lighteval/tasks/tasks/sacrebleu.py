"""
name:
Sacrebleu

dataset:
lighteval/sacrebleu_manual, wmt14, wmt16

abstract:
tasks from sacrebleu

languages:
english, german, french, japanese, korean, chinese, arabic

tags:
translation

paper:
https://github.com/mjpost/sacrebleu
"""

import ast

import pycountry

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


def __wmt_prompt(line, alphabetical, task_name: str = None):
    def language(code):
        # key is alpha_2 or alpha_3 depending on the code length
        language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
        return language_tuple.name

    # It would be better to just reupload the file tbh
    if isinstance(line["translation"], str):
        line["translation"] = ast.literal_eval(line["translation"])
        for k, v in line["translation"].items():
            line["translation"][k] = as_list(v)[0]

    l_in, l_out = sorted(line["translation"].keys(), reverse=not alphabetical)

    return Doc(
        task_name=task_name,
        query=f"{language(l_in)} phrase: " + line["translation"][l_in].rstrip() + f"\n{language(l_out)} phrase:",
        gold_index=0,
        choices=[line["translation"][l_out].rstrip()],
        instruction=f"Translate {language(l_in)} to {language(l_out)}, do not explain, only output the translation.",
    )


def wmt_alphabetical_prompt(line, task_name: str = None):
    return __wmt_prompt(line, True, task_name)


def wmt_reverse_alphabetical_prompt(line, task_name: str = None):
    return __wmt_prompt(line, False, task_name)


iwslt17_ar_en = LightevalTaskConfig(
    name="iwslt17:ar-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_ar-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_de_en = LightevalTaskConfig(
    name="iwslt17:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_en_ar = LightevalTaskConfig(
    name="iwslt17:en-ar",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_ar-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_en_de = LightevalTaskConfig(
    name="iwslt17:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_en_fr = LightevalTaskConfig(
    name="iwslt17:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_en_ja = LightevalTaskConfig(
    name="iwslt17:en-ja",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_en-ja",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_en_ko = LightevalTaskConfig(
    name="iwslt17:en-ko",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_en-ko",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_en_zh = LightevalTaskConfig(
    name="iwslt17:en-zh",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_en-zh",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_fr_en = LightevalTaskConfig(
    name="iwslt17:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_ja_en = LightevalTaskConfig(
    name="iwslt17:ja-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_ja-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_ko_en = LightevalTaskConfig(
    name="iwslt17:ko-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_ko-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

iwslt17_zh_en = LightevalTaskConfig(
    name="iwslt17:zh-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="iwslt17_zh-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

mtnt2019_en_fr = LightevalTaskConfig(
    name="mtnt2019:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="mtnt2019_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=200,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

mtnt2019_en_ja = LightevalTaskConfig(
    name="mtnt2019:en-ja",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="mtnt2019_en-ja",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=200,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

mtnt2019_fr_en = LightevalTaskConfig(
    name="mtnt2019:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="mtnt2019_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=200,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

mtnt2019_ja_en = LightevalTaskConfig(
    name="mtnt2019:ja-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="mtnt2019_ja-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=200,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_cs_en = LightevalTaskConfig(
    name="wmt08:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_de_en = LightevalTaskConfig(
    name="wmt08:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_en_cs = LightevalTaskConfig(
    name="wmt08:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_en_de = LightevalTaskConfig(
    name="wmt08:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_en_es = LightevalTaskConfig(
    name="wmt08:en-es",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_en-es",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_en_fr = LightevalTaskConfig(
    name="wmt08:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_en_hu = LightevalTaskConfig(
    name="wmt08:en-hu",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_en-hu",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_es_en = LightevalTaskConfig(
    name="wmt08:es-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_es-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_fr_en = LightevalTaskConfig(
    name="wmt08:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt08_hu_en = LightevalTaskConfig(
    name="wmt08:hu-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt08_hu-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_cs_en = LightevalTaskConfig(
    name="wmt09:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_de_en = LightevalTaskConfig(
    name="wmt09:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_en_cs = LightevalTaskConfig(
    name="wmt09:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_en_de = LightevalTaskConfig(
    name="wmt09:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_en_es = LightevalTaskConfig(
    name="wmt09:en-es",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_en-es",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_en_fr = LightevalTaskConfig(
    name="wmt09:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_en_hu = LightevalTaskConfig(
    name="wmt09:en-hu",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_en-hu",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_en_it = LightevalTaskConfig(
    name="wmt09:en-it",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_en-it",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_es_en = LightevalTaskConfig(
    name="wmt09:es-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_es-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_fr_en = LightevalTaskConfig(
    name="wmt09:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_hu_en = LightevalTaskConfig(
    name="wmt09:hu-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_hu-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt09_it_en = LightevalTaskConfig(
    name="wmt09:it-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt09_it-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_cs_en = LightevalTaskConfig(
    name="wmt10:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_de_en = LightevalTaskConfig(
    name="wmt10:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_en_cs = LightevalTaskConfig(
    name="wmt10:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_en_de = LightevalTaskConfig(
    name="wmt10:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_en_es = LightevalTaskConfig(
    name="wmt10:en-es",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_en-es",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_en_fr = LightevalTaskConfig(
    name="wmt10:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_es_en = LightevalTaskConfig(
    name="wmt10:es-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_es-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt10_fr_en = LightevalTaskConfig(
    name="wmt10:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt10_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_cs_en = LightevalTaskConfig(
    name="wmt11:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_de_en = LightevalTaskConfig(
    name="wmt11:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_en_cs = LightevalTaskConfig(
    name="wmt11:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_en_de = LightevalTaskConfig(
    name="wmt11:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_en_es = LightevalTaskConfig(
    name="wmt11:en-es",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_en-es",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_en_fr = LightevalTaskConfig(
    name="wmt11:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_es_en = LightevalTaskConfig(
    name="wmt11:es-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_es-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt11_fr_en = LightevalTaskConfig(
    name="wmt11:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt11_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_cs_en = LightevalTaskConfig(
    name="wmt12:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_de_en = LightevalTaskConfig(
    name="wmt12:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_en_cs = LightevalTaskConfig(
    name="wmt12:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_en_de = LightevalTaskConfig(
    name="wmt12:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_en_es = LightevalTaskConfig(
    name="wmt12:en-es",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_en-es",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_en_fr = LightevalTaskConfig(
    name="wmt12:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_es_en = LightevalTaskConfig(
    name="wmt12:es-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_es-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt12_fr_en = LightevalTaskConfig(
    name="wmt12:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt12_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_cs_en = LightevalTaskConfig(
    name="wmt13:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_de_en = LightevalTaskConfig(
    name="wmt13:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_en_cs = LightevalTaskConfig(
    name="wmt13:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_en_de = LightevalTaskConfig(
    name="wmt13:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_en_es = LightevalTaskConfig(
    name="wmt13:en-es",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_en-es",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_en_fr = LightevalTaskConfig(
    name="wmt13:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_en_ru = LightevalTaskConfig(
    name="wmt13:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_es_en = LightevalTaskConfig(
    name="wmt13:es-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_es-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_fr_en = LightevalTaskConfig(
    name="wmt13:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt13_ru_en = LightevalTaskConfig(
    name="wmt13:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt13_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_cs_en = LightevalTaskConfig(
    name="wmt14:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_de_en = LightevalTaskConfig(
    name="wmt14:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_en_cs = LightevalTaskConfig(
    name="wmt14:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_en_de = LightevalTaskConfig(
    name="wmt14:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_en_fr = LightevalTaskConfig(
    name="wmt14:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="wmt14",
    hf_subset="fr-en",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_en_fr = LightevalTaskConfig(
    name="wmt14:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_en_hi = LightevalTaskConfig(
    name="wmt14:en-hi",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_en-hi",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_en_ru = LightevalTaskConfig(
    name="wmt14:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_fr_en = LightevalTaskConfig(
    name="wmt14:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="wmt14",
    hf_subset="fr-en",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_fr_en = LightevalTaskConfig(
    name="wmt14:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_hi_en = LightevalTaskConfig(
    name="wmt14:hi-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_hi-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt14_ru_en = LightevalTaskConfig(
    name="wmt14:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt14_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_cs_en = LightevalTaskConfig(
    name="wmt15:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_de_en = LightevalTaskConfig(
    name="wmt15:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_en_cs = LightevalTaskConfig(
    name="wmt15:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_en_de = LightevalTaskConfig(
    name="wmt15:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_en_fi = LightevalTaskConfig(
    name="wmt15:en-fi",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_en-fi",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_en_fr = LightevalTaskConfig(
    name="wmt15:en-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_en-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_en_ru = LightevalTaskConfig(
    name="wmt15:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_fi_en = LightevalTaskConfig(
    name="wmt15:fi-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_fi-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_fr_en = LightevalTaskConfig(
    name="wmt15:fr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_fr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt15_ru_en = LightevalTaskConfig(
    name="wmt15:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt15_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_cs_en = LightevalTaskConfig(
    name="wmt16:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_de_en = LightevalTaskConfig(
    name="wmt16:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="wmt16",
    hf_subset="de-en",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_de_en = LightevalTaskConfig(
    name="wmt16:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_cs = LightevalTaskConfig(
    name="wmt16:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_de = LightevalTaskConfig(
    name="wmt16:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="wmt16",
    hf_subset="de-en",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_de = LightevalTaskConfig(
    name="wmt16:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_fi = LightevalTaskConfig(
    name="wmt16:en-fi",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_en-fi",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_ro = LightevalTaskConfig(
    name="wmt16:en-ro",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="wmt16",
    hf_subset="ro-en",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_ro = LightevalTaskConfig(
    name="wmt16:en-ro",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_en-ro",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_ru = LightevalTaskConfig(
    name="wmt16:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_en_tr = LightevalTaskConfig(
    name="wmt16:en-tr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_en-tr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_fi_en = LightevalTaskConfig(
    name="wmt16:fi-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_fi-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_ro_en = LightevalTaskConfig(
    name="wmt16:ro-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="wmt16",
    hf_subset="ro-en",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_ro_en = LightevalTaskConfig(
    name="wmt16:ro-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_ro-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_ru_en = LightevalTaskConfig(
    name="wmt16:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt16_tr_en = LightevalTaskConfig(
    name="wmt16:tr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt16_tr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_cs_en = LightevalTaskConfig(
    name="wmt17:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_de_en = LightevalTaskConfig(
    name="wmt17:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_en_cs = LightevalTaskConfig(
    name="wmt17:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_en_de = LightevalTaskConfig(
    name="wmt17:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_en_fi = LightevalTaskConfig(
    name="wmt17:en-fi",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_en-fi",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_en_lv = LightevalTaskConfig(
    name="wmt17:en-lv",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_en-lv",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_en_ru = LightevalTaskConfig(
    name="wmt17:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_en_tr = LightevalTaskConfig(
    name="wmt17:en-tr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_en-tr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_en_zh = LightevalTaskConfig(
    name="wmt17:en-zh",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_en-zh",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_fi_en = LightevalTaskConfig(
    name="wmt17:fi-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_fi-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_lv_en = LightevalTaskConfig(
    name="wmt17:lv-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_lv-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_ru_en = LightevalTaskConfig(
    name="wmt17:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_tr_en = LightevalTaskConfig(
    name="wmt17:tr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_tr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt17_zh_en = LightevalTaskConfig(
    name="wmt17:zh-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt17_zh-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_cs_en = LightevalTaskConfig(
    name="wmt18:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_de_en = LightevalTaskConfig(
    name="wmt18:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_en_cs = LightevalTaskConfig(
    name="wmt18:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_en_de = LightevalTaskConfig(
    name="wmt18:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_en_et = LightevalTaskConfig(
    name="wmt18:en-et",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_en-et",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_en_fi = LightevalTaskConfig(
    name="wmt18:en-fi",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_en-fi",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_en_ru = LightevalTaskConfig(
    name="wmt18:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_en_tr = LightevalTaskConfig(
    name="wmt18:en-tr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_en-tr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_en_zh = LightevalTaskConfig(
    name="wmt18:en-zh",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_en-zh",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_et_en = LightevalTaskConfig(
    name="wmt18:et-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_et-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_fi_en = LightevalTaskConfig(
    name="wmt18:fi-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_fi-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_ru_en = LightevalTaskConfig(
    name="wmt18:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_tr_en = LightevalTaskConfig(
    name="wmt18:tr-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_tr-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt18_zh_en = LightevalTaskConfig(
    name="wmt18:zh-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt18_zh-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_cs_de = LightevalTaskConfig(
    name="wmt19:cs-de",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_cs-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_de_cs = LightevalTaskConfig(
    name="wmt19:de-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_de-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_de_en = LightevalTaskConfig(
    name="wmt19:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_de_fr = LightevalTaskConfig(
    name="wmt19:de-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_de-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_cs = LightevalTaskConfig(
    name="wmt19:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_de = LightevalTaskConfig(
    name="wmt19:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_fi = LightevalTaskConfig(
    name="wmt19:en-fi",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-fi",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_gu = LightevalTaskConfig(
    name="wmt19:en-gu",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-gu",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_kk = LightevalTaskConfig(
    name="wmt19:en-kk",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-kk",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_lt = LightevalTaskConfig(
    name="wmt19:en-lt",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-lt",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_ru = LightevalTaskConfig(
    name="wmt19:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_en_zh = LightevalTaskConfig(
    name="wmt19:en-zh",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_en-zh",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_fi_en = LightevalTaskConfig(
    name="wmt19:fi-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_fi-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_fr_de = LightevalTaskConfig(
    name="wmt19:fr-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_fr-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_gu_en = LightevalTaskConfig(
    name="wmt19:gu-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_gu-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_kk_en = LightevalTaskConfig(
    name="wmt19:kk-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_kk-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_lt_en = LightevalTaskConfig(
    name="wmt19:lt-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_lt-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_ru_en = LightevalTaskConfig(
    name="wmt19:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt19_zh_en = LightevalTaskConfig(
    name="wmt19:zh-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt19_zh-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_cs_en = LightevalTaskConfig(
    name="wmt20:cs-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_cs-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_de_en = LightevalTaskConfig(
    name="wmt20:de-en",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_de-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_de_fr = LightevalTaskConfig(
    name="wmt20:de-fr",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_de-fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_cs = LightevalTaskConfig(
    name="wmt20:en-cs",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-cs",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_de = LightevalTaskConfig(
    name="wmt20:en-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_iu = LightevalTaskConfig(
    name="wmt20:en-iu",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-iu",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_ja = LightevalTaskConfig(
    name="wmt20:en-ja",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-ja",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_km = LightevalTaskConfig(
    name="wmt20:en-km",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-km",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_pl = LightevalTaskConfig(
    name="wmt20:en-pl",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-pl",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_ps = LightevalTaskConfig(
    name="wmt20:en-ps",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-ps",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_ru = LightevalTaskConfig(
    name="wmt20:en-ru",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_ta = LightevalTaskConfig(
    name="wmt20:en-ta",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-ta",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_en_zh = LightevalTaskConfig(
    name="wmt20:en-zh",
    prompt_function=wmt_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-zh",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_fr_de = LightevalTaskConfig(
    name="wmt20:fr-de",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_fr-de",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_iu_en = LightevalTaskConfig(
    name="wmt20:iu-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_iu-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_ja_en = LightevalTaskConfig(
    name="wmt20:ja-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_ja-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_km_en = LightevalTaskConfig(
    name="wmt20:km-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_km-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_pl_en = LightevalTaskConfig(
    name="wmt20:pl-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_pl-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_ps_en = LightevalTaskConfig(
    name="wmt20:ps-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_ps-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_ru_en = LightevalTaskConfig(
    name="wmt20:ru-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_ru-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_ta_en = LightevalTaskConfig(
    name="wmt20:ta-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_ta-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

wmt20_zh_en = LightevalTaskConfig(
    name="wmt20:zh-en",
    prompt_function=wmt_reverse_alphabetical_prompt,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_zh-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    wmt14_de_en,
    wmt16_en_cs,
    wmt19_en_cs,
    wmt19_en_de,
    wmt19_en_fi,
    wmt19_en_gu,
    wmt19_en_kk,
    wmt19_en_lt,
    wmt19_en_ru,
    wmt19_en_zh,
    wmt19_fi_en,
    wmt19_fr_de,
    wmt19_gu_en,
    wmt19_kk_en,
    wmt19_lt_en,
    wmt19_ru_en,
    wmt19_zh_en,
    wmt20_cs_en,
    wmt20_de_en,
    wmt20_en_de,
    wmt20_en_iu,
    wmt20_en_ja,
    wmt20_en_km,
    wmt20_en_pl,
    wmt20_en_ps,
    wmt20_en_ru,
    wmt20_en_ta,
    wmt20_en_zh,
    wmt20_fr_de,
    wmt20_iu_en,
    wmt20_ja_en,
    wmt20_km_en,
    wmt20_pl_en,
    wmt20_ps_en,
    wmt20_ru_en,
    wmt20_ta_en,
    wmt20_zh_en,
]

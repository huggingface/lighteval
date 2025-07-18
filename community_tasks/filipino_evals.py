# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401

"""
This file contains the tasks for the Filipino language, collectively known as FilBench.
It includes several tasks for the following categories: Cultural Knowledge, Classical NLP, Reading Comprehension, and Generation.
For more information, please read the paper: https://github.com/filbench/filbench-eval/blob/main/filbench.pdf

Contact:
- Lester James V. Miranda <ljvmiranda@gmail.com>
- Elyanah Aco <elyanah.aco02@gmail.com>
- Conner Manuel <manuel.conner.g@berkeley.edu>
- Jan Christian Blaise Cruz <jcbcruz02@gmail.com>
- Joseph Imperial <jmri20@bath.ac.uk>
"""

from collections import OrderedDict
from functools import partial
from typing import Any

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import (
    LogProbCharNorm,
    LogProbPMINorm,
    LogProbTokenNorm,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks import MMLU_SUBSETS
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro


# Balita NLP
FILIPINO_BALITA_TASKS = [
    LightevalTaskConfig(
        name=f"balita_tgl_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            language=Language.TAGALOG,
            adapter=lambda line: {
                "question": "Alin sa mga titlulong nakalista sa ibaba ang pinaka-angkop para sa teksto?",
                "context": f"Teksto: {line['title_choice_first_paragraph']}",
                "choices": line["title_choices"],
                "gold_idx": line["title_choice_gold_idx"],
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_repo="LanceBunag/BalitaNLP",
        hf_subset="no-image",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=("validation", "test"),
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# Belebele
FILIPINO_BELEBELE_TASKS = [
    LightevalTaskConfig(
        name=f"belebele_{LangCodeLanguage.get(language).to_alpha3()}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language).to_alpha3()],
            lambda line: {
                "question": line["question"],
                "context": line["flores_passage"],
                "choices": [line[f"mc_answer{i}"] for i in range(1, 5)],
                "gold_idx": int(line["correct_answer_num"]) - 1,
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_repo="facebook/belebele",
        hf_subset=language,
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
    for language in ["tgl_Latn", "ceb_Latn"]
]

# CebuaNER
cebuaner_choices = ["PERSON", "ORGANIZATION", "LOCATION", "OTHER"]
cebuaner_answer_idx = ["A", "B", "C", "D"]
question = "Unsa ang ginganlan nga named-entity sa pulong '{entity}' niini nga sentence: {text}"
FILIPINO_CEBUANER_TASKS = [
    LightevalTaskConfig(
        name=f"cebuaner_ceb_{formulation.name.lower()}",
        hf_subset="default",
        prompt_function=get_mcq_prompt_function(
            Language.CEBUANO,
            lambda line: {
                "question": question.format(entity=line["entity"], text=line["text"]),
                "choices": cebuaner_choices,
                "gold_idx": cebuaner_answer_idx.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="UD-Filipino/cebuaner-instruction",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        suite=["community"],
        generation_size=-1,
        trust_dataset=True,
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# Cebuano Readability
cebuano_readability_choices = ["Grade 1", "Grade 2", "Grade 3"]
cebuano_readability_instruction = """
Unsa ang angay nga lebel sa grado alang sa mosunod nga teksto?

Grade 1 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 6-7.
Grade 2 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 7-8.
Grade 3 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 8-9.
"""
FILIPINO_READABILITY_TASKS = [
    LightevalTaskConfig(
        name=f"readability_ceb_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.CEBUANO,
            lambda line: {
                "question": cebuano_readability_instruction + line["text"],
                "choices": cebuano_readability_choices,
                "gold_idx": cebuano_readability_choices.index(f"Grade {line['label']}"),
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_subset="default",
        hf_repo="UD-Filipino/cebuano-readability",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# Dengue
dengue_filipino_subsets = {
    "absent": "pagiging absent",
    "dengue": "dengue",
    "health": "kalusugan",
    "mosquito": "lamok",
    "sick": "sakit",
}


def filipino_dengue_pfn(line, task_name: str) -> Doc:
    subset = task_name.split(":")[-1]
    subset_keyword = dengue_filipino_subsets[subset]

    instruction = f"Tungkol ba sa {subset_keyword} ang sumusunod na pangungusap? Piliin ang tamang sagot:\n\n"
    choices: dict[str, str] = OrderedDict({"A": "Hindi", "B": "Oo"})

    answer_index = int(line.get(subset))
    query = f"{instruction}{line['text']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in choices.items()])
    query += "Sagot:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(choices.keys()),
        gold_index=answer_index,
        instruction=instruction,
    )


FILIPINO_DENGUE_TASKS = [
    LightevalTaskConfig(
        name=f"dengue_filipino_fil:{subset}",
        hf_subset="default",
        prompt_function=filipino_dengue_pfn,
        hf_repo="jcblaise/dengue_filipino",
        metrics=[Metrics.loglikelihood_acc_norm],
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random",
        suite=("community",),
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for subset in dengue_filipino_subsets
]

# FireCS
firecs_choices = ["Negatibo", "Neutral", "Positibo"]

FILIPINO_FIRECS_TASK = [
    LightevalTaskConfig(
        name=f"firecs_fil_{formulation.name.lower()}",
        hf_subset="default",
        prompt_function=get_mcq_prompt_function(
            Language.TAGALOG,
            lambda line: {
                "question": f"Ano ang damdamin o sentimiyento ng sumusunod na pangungusap: {line['review']}",
                "choices": firecs_choices,
                "gold_idx": int(line["label"]),
            },
        ),
        hf_repo="ccosme/FiReCS",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["train", "test"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random",
        suite=["community"],
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# Global-MMLU (FIl)

FILIPINO_GLOBAL_MMLU_TASKS = [
    LightevalTaskConfig(
        name=f"global_mmlu_{sensitivity_label.lower()}_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [
                    line["option_a"],
                    line["option_b"],
                    line["option_c"],
                    line["option_d"],
                ],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=partial(
            lambda subset, sensitivity_label, x: x["subject"].lower() == subset
            and (
                sensitivity_label == "ALL" or sensitivity_label in x["cultural_sensitivity_label"].replace("-", "UNK")
            ),
            subset,
            sensitivity_label,
        ),
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for language in [Language.TAGALOG]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
    for sensitivity_label in ["ALL", "CA", "CS", "UNK"]
]

# INCLUDE

FILIPINO_INCLUDE_TASKS = [
    LightevalTaskConfig(
        name=f"include_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [line[f"option_{i}"] for i in ("a", "b", "c", "d")],
                "gold_idx": line["answer"],
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_subset="Tagalog",
        hf_repo="CohereForAI/include-base-44",
        hf_filter=partial(lambda subset, x: x["subject"].replace(" ", "_").lower() == subset, subset),
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for subset in ["culturology", "history", "language", "driving_license"]
    for language in [Language.TAGALOG]
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# KALAHI
FILIPINO_KALAHI_TASKS = [
    LightevalTaskConfig(
        name=f"kalahi_tgl_{formulation.name.lower()}",
        suite=["community"],
        prompt_function=get_mcq_prompt_function(
            language=Language.TAGALOG,
            adapter=lambda line: {
                "question": line["prompts"][0]["question"],
                "choices": [entry[3:] for entry in line["prompts"][0]["mcq"].split("\n")],
                "gold_idx": LETTER_INDICES.index(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="aisingapore/cultural_evaluation-kalahi",
        hf_subset="default",
        evaluation_splits=["tl"],
        metrics=[
            loglikelihood_acc_metric(normalization=None),
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbCharNorm()),
        ],
    )
    for formulation in [HybridFormulation(), MCFFormulation()]
]

# NewsPH NLI
FILIPINO_NEWSPH_NLI_TASKS = [
    LightevalTaskConfig(
        name=f"newsphnli_fil_{formulation.name.lower()}",
        suite=["community"],
        prompt_function=get_nli_prompt_function(
            language=Language.TAGALOG,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since there is no neutral label
                "gold_idx": line["label"],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_repo="jcblaise/newsph_nli",
        hf_subset="default",
        evaluation_splits=["validation"],
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
        trust_dataset=True,
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# NTREX-128
FILIPINO_NTREX_TASK = [
    LightevalTaskConfig(
        name=f"ntrex128_{LangCodeLanguage.get(language).to_alpha3()}",
        prompt_function=get_translation_prompt_function(
            source_language=Language.ENGLISH,
            target_language=iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language).to_alpha3()],
            adapter=lambda line: {
                "source_text": line["eng_Latn"],
                "target_text": line[language],
            },
            formulation=CFFormulation(),
        ),
        suite=("community",),
        hf_repo="mteb/NTREX",
        hf_subset="default",
        metrics=[
            Metrics.rougeL,
            Metrics.bleu,
            Metrics.bleurt,
            Metrics.chrf,
            Metrics.ter,
        ],
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=64,
        trust_dataset=True,
        version=0,
    )
    for language in ["fil_Latn"]
]

# SIB-200

sib200_choices = [
    "geography",
    "science/technology",
    "entertainment",
    "travel",
    "sports",
    "health",
    "politics",
]


def get_instruction(language: Language) -> str:
    if language == Language.CEBUANO:
        return "Mahitungod sa unsa ang mosunod nga teksto?\n"
    if language == Language.TAGALOG:
        return "Tungkol saan ang sumusunod na pangungusap?\n"


def create_sib200_task(language: Language, formulation):
    return LightevalTaskConfig(
        name=f"sib200_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": get_instruction(language) + line["text"],
                "choices": sib200_choices,
                "gold_idx": sib200_choices.index(line["category"]),
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_subset=f"{language.value}_Latn",
        hf_repo="Davlan/sib200",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="validation",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )


FILIPINO_SIB_TASKS = [
    create_sib200_task(language, formulation)
    for language in [Language.TAGALOG, Language.CEBUANO]
    for formulation in [MCFFormulation(), HybridFormulation()]
]


def prepare_stingray_correctness(line: dict[str, str]) -> dict[str, Any]:
    # lang2 is Tagalog
    word = line["word"]
    sentence = line["lang2_sentence"]
    question = f"Is the usage of {word} in this sentence correct? \n{sentence}"
    choices = ["Yes", "No"]
    gold_idx = choices.index(line["usage_correctness_lang2_answer"])
    return {"question": question, "choices": choices, "gold_idx": gold_idx}


def prepare_stingray_semantic_appropriateness(line: dict[str, str]) -> dict[str, Any]:
    lang1 = line["lang1_sentence"]
    lang2 = line["lang2_sentence"]
    question = "Which sentence is more semantically appropriate?"
    choices = [lang1, lang2, "Both"]
    choice_letters = ["A", "B", "C"]
    gold_idx = choice_letters.index(line["semantic_appropriate_answer"])
    return {"question": question, "choices": choices, "gold_idx": gold_idx}


FILIPINO_STINGRAY_CORRECTNESS_TASKS = [
    LightevalTaskConfig(
        name=f"stingraybench_correctness_tgl_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,  # the orig instruction is in English, so we replicate it.
            adapter=prepare_stingray_correctness,
            formulation=formulation,
        ),
        suite=("community",),
        hf_subset="id_tl",
        hf_repo="StingrayBench/StingrayBench",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

FILIPINO_STINGRAY_SEMANTIC_TASKS = [
    LightevalTaskConfig(
        name=f"stingraybench_semantic_appropriateness_tgl_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,  # the orig instruction is in English, so we replicate it.
            adapter=prepare_stingray_semantic_appropriateness,
            formulation=formulation,
        ),
        suite=("community",),
        hf_subset="id_tl",
        hf_repo="StingrayBench/StingrayBench",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

FILIPINO_STINGRAY_TASKS = FILIPINO_STINGRAY_SEMANTIC_TASKS + FILIPINO_STINGRAY_CORRECTNESS_TASKS

# Tatoeba
# We follow the original translation direction from tatoeba
lang_dict = {
    "ceb": {
        "subset": "ceb-eng",
        "source_language": Language.CEBUANO,
        "target_language": Language.ENGLISH,
    },
    "tgl": {
        "subset": "eng-tgl",
        "source_language": Language.ENGLISH,
        "target_language": Language.TAGALOG,
    },
}

FILIPINO_TATOEBA_TASKS = [
    LightevalTaskConfig(
        name=f"tatoeba_{language}",
        prompt_function=get_translation_prompt_function(
            source_language=meta.get("source_language"),
            target_language=meta.get("target_language"),
            adapter=lambda line: {
                "source_text": line["sourceString"],
                "target_text": line["targetString"],
            },
            formulation=CFFormulation(),
        ),
        suite=("community",),
        hf_repo="Helsinki-NLP/tatoeba_mt",
        hf_subset=meta.get("subset"),
        metrics=[
            Metrics.rougeL,
            Metrics.bleu,
            Metrics.bleurt,
            Metrics.chrf,
            Metrics.ter,
        ],
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        trust_dataset=True,
        generation_size=64,
    )
    for language, meta in lang_dict.items()
]

# TICO-19
FILIPINO_TICO19_TASKS = [
    LightevalTaskConfig(
        name="tico19_tgl",
        prompt_function=get_translation_prompt_function(
            source_language=Language.ENGLISH,
            target_language=Language.TAGALOG,
            adapter=lambda line: {
                "source_text": line["sourceString"],
                "target_text": line["targetString"],
            },
            formulation=CFFormulation(),
        ),
        suite=("community",),
        hf_repo="gmnlp/tico19",
        hf_subset="en-tl",
        metrics=[
            Metrics.rougeL,
            Metrics.bleu,
            Metrics.bleurt,
            Metrics.chrf,
            Metrics.ter,
        ],
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=["validation"],
        few_shots_select="random",
        trust_dataset=True,
        generation_size=64,
    )
]

# TLUnified-NER
tlunified_ner_choices = ["PERSON", "ORGANIZATION", "LOCATION"]
tlunified_ner_answer_idx = ["A", "B", "C"]

FILIPINO_TLUNIFIED_NER_TASK = [
    LightevalTaskConfig(
        name=f"tlunifiedner_tgl_{formulation.name.lower()}",
        hf_subset="instruction",
        prompt_function=get_mcq_prompt_function(
            Language.TAGALOG,
            lambda line: {
                "question": f"Ano ang named-entity ng salitang '{line['entity']}' sa pangungusap na ito: {line['text']}",
                "choices": tlunified_ner_choices,
                "gold_idx": tlunified_ner_answer_idx.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="ljvmiranda921/tlunified-ner",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        suite=["community"],
        generation_size=-1,
        trust_dataset=True,
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# Universal NER
universalner_choices = ["PERSON", "ORGANIZATION", "LOCATION"]
universalner_answer_idx = ["A", "B", "C"]


def create_universalner_task(language: Language, formulation):
    if language == Language.CEBUANO:
        question = "Unsa ang ginganlan nga named-entity sa pulong '{entity}' niini nga sentence: {text}"
    if language == Language.TAGALOG:
        question = "Ano ang named-entity ng salitang '{entity}' sa pangungusap na ito: {text}"

    return LightevalTaskConfig(
        name=f"universalner_{language.value}_{formulation.name.lower()}",
        hf_subset=language.value,
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": question.format(entity=line["entity"], text=line["text"]),
                "choices": universalner_choices,
                "gold_idx": universalner_answer_idx.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="UD-Filipino/universalner-instruction",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        suite=["community"],
        generation_size=-1,
        trust_dataset=True,
        metrics=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        version=0,
    )


FILIPINO_UNIVERSALNER_TASKS = [
    create_universalner_task(language, formulation)
    for language in [Language.CEBUANO, Language.TAGALOG]
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# Tasks Table

TASKS_TABLE: list[LightevalTaskConfig] = (
    FILIPINO_BALITA_TASKS
    + FILIPINO_BELEBELE_TASKS
    + FILIPINO_CEBUANER_TASKS
    + FILIPINO_READABILITY_TASKS
    + FILIPINO_DENGUE_TASKS
    + FILIPINO_FIRECS_TASK
    + FILIPINO_GLOBAL_MMLU_TASKS
    + FILIPINO_INCLUDE_TASKS
    + FILIPINO_KALAHI_TASKS
    + FILIPINO_NEWSPH_NLI_TASKS
    + FILIPINO_NTREX_TASK
    + FILIPINO_SIB_TASKS
    + FILIPINO_STINGRAY_TASKS
    + FILIPINO_TATOEBA_TASKS
    + FILIPINO_TICO19_TASKS
    + FILIPINO_TLUNIFIED_NER_TASK
    + FILIPINO_UNIVERSALNER_TASKS
)

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
Custom evaluation tasks for lighteval

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import random
import re
from typing import Any, Dict, List, Optional, Union

from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import Metric, MetricCategory, Metrics
from lighteval.metrics.utils.metric_utils import MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]
# fmt: on

# ArabicMMLU
# fmt: off
ARABIC_MMLU_SUBSETS = [
    "All", "Islamic Studies", "Islamic Studies (Middle School)", "Islamic Studies (Primary School)", "Islamic Studies (High School)", "Driving Test",
    "Natural Science (Middle School)", "Natural Science (Primary School)", "History (Middle School)", "History (Primary School)", "History (High School)", "General Knowledge",
    "General Knowledge (Middle School)", "General Knowledge (Primary School)", "Law (Professional)", "Physics (High School)", "Social Science (Middle School)",
    "Social Science (Primary School)", "Management (University)", "Arabic Language (Middle School)", "Arabic Language (Primary School)", "Arabic Language (High School)", "Political Science (University)",
    "Philosophy (High School)", "Accounting (University)", "Computer Science (Middle School)", "Computer Science (Primary School)", "Computer Science (High School)", "Computer Science (University)",
    "Geography (Middle School)", "Geography (Primary School)", "Geography (High School)", "Math (Primary School)", "Biology (High School)", "Economics (Middle School)",
    "Economics (High School)", "Economics (University)", "Arabic Language (General)", "Arabic Language (Grammar)", "Civics (Middle School)", "Civics (High School)"
]
# fmt: on


def arabic_mmlu_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"

    # Define the mapping from Latin to Arabic letters
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}

    # Create a list of valid choices with corresponding Arabic keys
    choices = []
    valid_keys_latin = []
    valid_keys_arabic = []

    # Enumerate through the options and append the valid ones
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:  # Check if option is not null
            choices.append(option)
            valid_keys_latin.append(key)  # Append the Latin key (A, B, C, D, E)
            valid_keys_arabic.append(latin_to_arabic[key])  # Append the corresponding Arabic letter

    # Find the correct index for the answer key in the Arabic version
    answer_index = valid_keys_latin.index(line["Answer Key"])

    # Construct the query with Arabic letters
    query = f"{instruction}{line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys_arabic, choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,  # Return only valid choices (Arabic keys)
        gold_index=answer_index,  # Correct index in the valid Arabic keys
        instruction=instruction,
    )


class CustomArabicMMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_pfn,
            hf_repo="MBZUAI/ArabicMMLU",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ARABIC_MMLU_TASKS = [
    CustomArabicMMLUTask(name=f"arabic_mmlu:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_SUBSETS
]


# ARABIC MMLU HT ##
# fmt: off
ARABIC_MMLU_HT_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]
# fmt: on


def arabic_mmlu_ht_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    choices = line["choices"]
    answer_index = line["answer"]  # It is an int reflecting the index of correct answer in line["choices"]

    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{idx}. {choice}\n" for idx, choice in enumerate(choices, start=1)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(i) for i in range(1, len(choices) + 1)],  # List of strings instead of ints
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUHTTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_ht_pfn,
            hf_repo="MBZUAI/human_translated_arabic_mmlu",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ARABIC_MMLU_HT_TASKS = [
    CustomArabicMMLUHTTask(name=f"arabic_mmlu_ht:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_HT_SUBSETS
]

# ARABIC MMLU MT ##
# fmt: off
ARABIC_MMLU_MT_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]
# fmt: on


def arabic_mmlu_mt_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة: أ، ب، ج، أو د... إلخ. \n\n"
    choices = [line["A"], line["B"], line["C"], line["D"]]
    # Answers are provided with roman letters - we look for the correct index in LETTER_INDICES,
    # it will then be applied to arabic letters
    answer_index = LETTER_INDICES.index(
        line["answer"]
    )  # line["answer"] is the correct answer. That's why we need to index it !

    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES_AR[:4], choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUMTTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_mt_pfn,
            hf_repo="OALL/Arabic_MMLU",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ARABIC_MMLU_MT_TASKS = [
    CustomArabicMMLUMTTask(name=f"arabic_mmlu_mt:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_MT_SUBSETS
]


# ACVA ##
# fmt: off
ACVA_SUBSETS = [
    "Algeria", "Ancient_Egypt", "Arab_Empire", "Arabic_Architecture", "Arabic_Art", "Arabic_Astronomy", "Arabic_Calligraphy", "Arabic_Ceremony",
    "Arabic_Clothing", "Arabic_Culture", "Arabic_Food", "Arabic_Funeral", "Arabic_Geography", "Arabic_History", "Arabic_Language_Origin",
    "Arabic_Literature", "Arabic_Math", "Arabic_Medicine", "Arabic_Music", "Arabic_Ornament", "Arabic_Philosophy", "Arabic_Physics_and_Chemistry",
    "Arabic_Wedding", "Bahrain", "Comoros", "Egypt_modern", "InfluenceFromAncientEgypt", "InfluenceFromByzantium", "InfluenceFromChina",
    "InfluenceFromGreece", "InfluenceFromIslam", "InfluenceFromPersia", "InfluenceFromRome", "Iraq", "Islam_Education", "Islam_branches_and_schools",
    "Islamic_law_system", "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania", "Mesopotamia_civilization", "Morocco", "Oman", "Palestine", "Qatar",
    "Saudi_Arabia", "Somalia", "Sudan", "Syria", "Tunisia", "United_Arab_Emirates", "Yemen",
    "communication", "computer_and_phone", "daily_life", "entertainment"
]
# fmt: on


def acva_pfn(line, task_name: str = None):
    question = line["question"]
    answer = line["answer"]

    return Doc(
        task_name=task_name,
        query=f"السؤال: {question}\nالإجابة:",
        choices=["صح", "خطأ"],
        gold_index=["صح", "خطأ"].index(answer),
    )


class CustomACVATask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=acva_pfn,
            hf_repo="OALL/ACVA",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ACVA_TASKS = [CustomACVATask(name=f"acva:{subset}", hf_subset=subset) for subset in ACVA_SUBSETS]


# AraTrust ##
# fmt: off
ARATRUST_SUBSETS = [
    "Trustfulness", "MentalHealth", "PhysicalHealth", "Offensive", "Ethics", "Privacy", "Unfairness", "Illegal",
]
# fmt: on


def aratrust_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة: أ، ب أو ج. \n\n"
    choices = [line["A"], line["B"], line["C"]]
    # Answers are provided with roman letters - we look for the correct index in LETTER_INDICES,
    # it will then be applied to arabic letters
    answer_index = LETTER_INDICES_AR.index(
        line["Answer"]
    )  # line["answer"] is the correct answer. That's why we need to index it !

    query = f"{instruction}{line['Question']}\n"
    query += "".join([f"{choice}\n" for choice in choices])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:3],
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAraTrustTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=aratrust_pfn,
            hf_repo="asas-ai/AraTrust-categorized",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ARATRUST_TASKS = [CustomAraTrustTask(name=f"aratrust:{subset}", hf_subset=subset) for subset in ARATRUST_SUBSETS]


def arabic_exams_pfn(line, task_name: str = None):
    topic = line["subject"]
    question = line["question"]
    choices = [line["A"], line["B"], line["C"], line["D"]]
    choices_formatted = [f" {LETTER_INDICES_AR[i]}) {choice}\n" for i, choice in enumerate(choices)]
    answer = line["answer"]
    answer_index = LETTER_INDICES.index(answer)

    instruction = f"الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول {topic.replace('_', ' ')}. \n\n"
    query = f"{instruction}السؤال: {question}\n"
    query += "\n".join(choices_formatted)
    query += "\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=answer_index,
        instruction=instruction,
    )


# ARABIC EXAMS ##
arabic_exams_task = LightevalTaskConfig(
    name="arabic_exams",
    prompt_function=arabic_exams_pfn,
    suite=["community"],
    hf_repo="OALL/Arabic_EXAMS",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# ALGHAFA NATIVE ##
# fmt: off
ALGHAFA_SUBSETS = [
    "mcq_exams_test_ar", "meta_ar_dialects", "meta_ar_msa", "multiple_choice_facts_truefalse_balanced_task", "multiple_choice_grounded_statement_soqal_task",
    "multiple_choice_grounded_statement_xglue_mlqa_task", "multiple_choice_rating_sentiment_no_neutral_task", "multiple_choice_rating_sentiment_task",
    "multiple_choice_sentiment_task"
]
# fmt: on


def alghafa_pfn(line, task_name: str = None):
    question = line["query"]
    answer_index = int(line["label"])
    allowed_keys = [f"sol{i}" for i in range(1, 6)]
    extracted_choices = [line[key] for key in allowed_keys if key in line]
    choices = [str(i) for i in range(len(extracted_choices))]

    instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
    query = f"{instruction}السؤال: {question}\n"

    for index, choice in enumerate(extracted_choices):
        query += f"{index}) {choice}\n"

    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAlGhafaNativeTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=alghafa_pfn,
            hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ALGHAFA_TASKS = [CustomAlGhafaNativeTask(name=f"alghafa:{subset}", hf_subset=subset) for subset in ALGHAFA_SUBSETS]

# ALGHAFA TRANSLATED ##
# race_ar
race_ar_task = LightevalTaskConfig(
    name="race_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="race_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# piqa_ar
piqa_ar_task = LightevalTaskConfig(
    name="piqa_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="piqa_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# arc_easy_ar
arc_easy_ar_task = LightevalTaskConfig(
    name="arc_easy_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="arc_easy_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# arc_challenge_okapi_ar
arc_challenge_okapi_ar_task = LightevalTaskConfig(
    name="arc_challenge_okapi_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="arc_challenge_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# mmlu_okapi_ar
mmlu_okapi_ar_task = LightevalTaskConfig(
    name="mmlu_okapi_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="mmlu_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# openbook_qa_ext_ar
openbook_qa_ext_ar_task = LightevalTaskConfig(
    name="openbook_qa_ext_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="openbook_qa_ext_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# boolq_ar
def boolq_arabic_pfn(line, task_name: str = None):
    question = line["question"]
    passage = line["passage"]
    instruction = "بناء على المقطع التالي، أجب عن السؤال ب نعم أو لا"
    query = f"""{instruction}
    المقطع :
    {passage}
    السؤال:
    {question}
    الإجابة:
    """

    return Doc(
        task_name=task_name,
        query=query,
        choices=["نعم", "لا"],
        gold_index=0 if line["answer"] else 1,
        instruction=instruction,
    )


boolq_ar_task = LightevalTaskConfig(
    name="boolq_ar",
    prompt_function=boolq_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="boolq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# copa_ext_ar
def copa_arabic_pfn(line, task_name: str = None):
    premise = line["premise"]
    choices = [line["choice1"], line["choice2"]]
    question_map = {"cause": "لأن", "effect": "لذلك"}
    question = question_map[line["question"]]
    answer = line["label"]

    query = "{}، {} :\n0) {}\n1) {}\nالإجابة:".format(premise, question, choices[0], choices[1])

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer,
        instruction="",
    )


copa_ext_ar_task = LightevalTaskConfig(
    name="copa_ext_ar",
    prompt_function=copa_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="copa_ext_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# hellaswag_okapi_ar
def hellaswag_arabic_pfn(line, task_name: str = None):
    ctx = re.sub(r"\[.*?\]", "", line["ctx"])  # Remove latin words within brackets
    endings = [
        re.sub(r"\[.*?\]", "", e) for e in eval(line["endings"])
    ]  # endings is a string representation of a list
    answer_index = line["label"]
    instruction = "بناء على السياق التالي، اختر النهاية الصحيحة من الاقتراحات التالية"

    query = f"""{instruction}
    السياق:
    {ctx}
    الاقتراحات:

    """
    for i, ending in enumerate(endings):
        query += f"{i}) {ending}\n"
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=endings,
        gold_index=answer_index,
        instruction=instruction,
    )


hellaswag_okapi_ar_task = LightevalTaskConfig(
    name="hellaswag_okapi_ar",
    prompt_function=hellaswag_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="hellaswag_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# toxigen_ar
def toxigen_arabic_pfn(line, task_name: str = None):
    text = line["text"]
    label = 1 if ((line["toxicity_ai"] + line["toxicity_human"]) > 5.5) else 0
    instruction = 'هل العبارة التالية تحض على الكراهية؟ قم بالرد بـ"نعم" أو "لا".'

    query = f"""{instruction}
    العبارة:
    '{text}'
    الإجابة:
    """

    return Doc(
        task_name=task_name,
        query=query,
        choices=["لا", "نعم"],
        gold_index=label,
        instruction=instruction,
    )


toxigen_ar_task = LightevalTaskConfig(
    name="toxigen_ar",
    prompt_function=toxigen_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="toxigen_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# sciq_ar
def sciq_arabic_pfn(line, task_name: str = None):
    support = line["support"]
    question = line["question"]
    correct_answer = line["correct_answer"]
    choices = [line["distractor1"], line["distractor2"], line["distractor3"], correct_answer]

    # Shuffle the choices
    random.shuffle(choices)

    answer_index = choices.index(correct_answer)

    instruction = "بناءً على السياق أدناه، اختر الإجابة الصحيحة للسؤال التالي من قائمة الاقتراحات"

    query = f"""{instruction}
    السياق:
    {support}
    السؤال:
    {question}
    الإجابات المحتملة:

    """
    for i, choice in enumerate(choices):
        query += f"{i}) {choice}\n"
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


sciq_ar_task = LightevalTaskConfig(
    name="sciq_ar",
    prompt_function=sciq_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="sciq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# madinah_qa
# fmt: off
MADINAH_QA_SUBSETS = ["Arabic Language (General)", "Arabic Language (Grammar)"]
# fmt: on


def madinah_qa_pfn(line, task_name: str = None):
    instruction = "بناءً على السياق أدناه، اختر الإجابة الصحيحة للسؤال التالي من قائمة الأجوبة:\n\n"

    # Define the mapping from Latin to Arabic letters
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}

    # Create a list of valid choices with corresponding Arabic keys
    choices = []
    valid_keys_latin = []
    valid_keys_arabic = []

    # Enumerate through the options and append the valid ones
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:  # Check if option is not null
            choices.append(option)
            valid_keys_latin.append(key)  # Append the Latin key (A, B, C, D, E)
            valid_keys_arabic.append(latin_to_arabic[key])  # Append the corresponding Arabic letter

    # Find the correct index for the answer key in the Arabic version
    answer_index = valid_keys_latin.index(line["Answer Key"])

    query = f"{instruction}\nالسياق:\n{line['Context']}\nالسؤال:\n{line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys_arabic, choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,
        gold_index=answer_index,  # Correct index in the valid keys
        instruction=instruction,
    )


class CustomMadinahQATask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=madinah_qa_pfn,
            hf_repo="MBZUAI/MadinahQA",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


MADINAH_QA_TASKS = [
    CustomMadinahQATask(name=f"madinah_qa:{subset}", hf_subset=subset) for subset in MADINAH_QA_SUBSETS
]


class JudgeMetricWrapper(Metric):
    """Wrapper class for LLM-based judge metric implementation."""

    def __init__(self, judge: JudgeLM):
        """
        Initializes the judge metric wrapper.

        Args:
            judge (JudgeLM): The LLM judge instance to use for evaluation.
        """
        self.judge = judge
        self.metric_name = "llm_as_judge"
        self.category = MetricCategory.LLM_AS_JUDGE
        self.corpus_level_fn = self.aggregate_scores
        self.sample_level_fn = self._sample_level_fn
        self.higher_is_better = True  # Fixed tuple syntax
        self.use_case = MetricUseCase.NONE

    def compute(self, responses: list[str], formatted_docs: list[Doc], **kwargs) -> dict[str, float]:
        """
        Computes evaluation scores using the judge's evaluate_answer method.

        Args:
            responses (list[str]): The predicted answers
            formatted_docs (list[Doc]): Documents containing questions and gold answers

        Returns:
            dict[str, float]: Dictionary containing evaluation scores
        """
        results = []
        for i, doc in enumerate(formatted_docs):
            question = doc.query
            gold = doc.choices[doc.gold_index] if doc.gold_index is not None else None
            answer = responses[i][0].result[0]

            score, _, _ = self.judge.evaluate_answer(question=question, answer=answer, options=None, gold=gold)
            results.append({self.metric_name: score})

        return results

    def aggregate_scores(self, scores: list[dict]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    def _sample_level_fn(self):
        return None


def parse_candidates(candidates: Union[List[str], str]) -> List[str]:
    """
    Parses and validates candidate answers from either list or string format.

    Args:
        candidates: Either a list of candidate answers or a newline-separated string

    Returns:
        List[str]: List of validated candidate answers

    Raises:
        ValueError: If candidates cannot be parsed or are empty
    """
    try:
        if isinstance(candidates, list):
            parsed_candidates = [str(c).strip() for c in candidates if c]
        else:
            parsed_candidates = [c.strip() for c in str(candidates).split("\n") if c.strip()]

        if not parsed_candidates:
            raise ValueError("No valid candidates found after parsing")

        return parsed_candidates
    except Exception as e:
        raise ValueError(f"Failed to parse candidates: {str(e)}")


def qa_prompt_arabic(line: Dict[str, Any], task_name: str = None) -> Doc:
    """
    Formats the prompt for Arabic question answering with candidates.

    Args:
        line: Dictionary containing question and candidate information
        task_name: Optional name for the task

    Returns:
        Doc: Formatted document for evaluation

    Raises:
        ValueError: If required fields are missing or invalid
    """
    try:
        # Validates and extracts the question
        if not isinstance(line.get("question"), str):
            raise ValueError("Question must be a string")
        question = line["question"]

        # Processes candidate answers
        candidates = parse_candidates(line["candidates"])

        # Validates gold answer
        if "gold_answer" not in line:
            raise ValueError("Gold answer is required")
        gold_answer = str(line["gold_answer"])

        # Constructs the prompt
        instruction = "بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي"
        query = f"{instruction}\n\nالسؤال:\n{question}\n\nالسياقات المقترحة:\n{', '.join(candidates)}\n"

        return Doc(
            task_name=task_name or "alrage",
            query=query,
            instruction=instruction,
            choices=[gold_answer],  # Gold answer is used as the only valid choice
            gold_index=0,  # Index of the correct answer in choices
        )
    except Exception as e:
        raise ValueError(f"Failed to create QA prompt: {str(e)}")


def judge_template(question: str, answer: str, gold: str, options: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Template for the Arabic judge prompt.

    System prompt translation:
    You are a neutral expert evaluator. Your tasks are:
    1. Evaluate the answer's accuracy compared to the correct answer
    2. Verify that the answer is supported by the provided context
    3. Evaluate the quality and comprehensiveness of the answer
    Rate the answer on a scale from 0 to 10.

    Args:
        question: The question being evaluated
        answer: The provided answer
        gold: The correct answer
        options: Optional list of answer choices

    Returns:
        List[Dict[str, str]]: Formatted messages for the judge
    """
    messages = [
        {
            "role": "system",
            "content": """أنت مقيّم محايد خبير باللغة العربية. يجب عليك:
1. تقييم دقة الإجابة مقارنة بالإجابة الصحيحة
2. التحقق من أن الإجابة مدعومة بالسياق المقدم
3. تقييم جودة وشمولية الإجابة

مهم جداً: يجب أن يكون ردك رقماً فقط من 0 إلى 10. لا تضف أي نص أو تفسير.""",
        },
        {
            "role": "user",
            "content": f"""السؤال: {question}

الإجابة المقدمة: {answer}

الإجابة الصحيحة: {gold}

أعط تقييماً من 0 إلى 10:
0-2: إجابة خاطئة تماماً
3-4: إجابة جزئية مع أخطاء
5-6: إجابة متوسطة
7-8: إجابة جيدة
9-10: إجابة ممتازة

اكتب رقماً فقط من 0 إلى 10 بدون أي نص إضافي:""",
        },
    ]
    return messages


def process_judge_response(response) -> float:
    """Process the judge's response to extract the score"""
    # If response is a list, extract the content from the user role
    if isinstance(response, list):
        response_content = " ".join(item["content"] for item in response if item["role"] == "user")
    else:
        response_content = response  # If it's not a list, use it directly

    try:
        # Extract the score from the response content
        score = float(next(num for num in response_content.split() if num.replace(".", "", 1).isdigit()))
        return min(max(score / 10.0, 0.0), 1.0)
    except (StopIteration, ValueError):
        return 0.0


judge = JudgeLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    templates=judge_template,
    process_judge_response=process_judge_response,
    judge_backend="vllm",
)

wrapped_judge = JudgeMetricWrapper(judge)

# Task configuration
alrage_qa_task = LightevalTaskConfig(
    name="alrage_qa",
    prompt_function=qa_prompt_arabic,
    suite=["community"],
    hf_repo="OALL/ALRAGE",
    hf_subset=None,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    metric=[wrapped_judge],
    trust_dataset=True,
    generation_size=200,
    stop_sequence=[],
    version=0,
)

TASKS_TABLE = (
    ARABIC_MMLU_TASKS
    + ARABIC_MMLU_HT_TASKS
    + ARABIC_MMLU_MT_TASKS
    + ACVA_TASKS
    + ALGHAFA_TASKS
    + ARATRUST_TASKS
    + MADINAH_QA_TASKS
    + [arabic_exams_task]
    + [race_ar_task]
    + [piqa_ar_task]
    + [arc_easy_ar_task]
    + [arc_challenge_okapi_ar_task]
    + [mmlu_okapi_ar_task]
    + [openbook_qa_ext_ar_task]
    + [boolq_ar_task]
    + [copa_ext_ar_task]
    + [hellaswag_okapi_ar_task]
    + [toxigen_ar_task]
    + [sciq_ar_task]
    + [alrage_qa_task]
)

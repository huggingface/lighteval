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

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.doc import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES
from lighteval.metrics.metrics import Metrics
from ..utils.metrics import get_qa_metric

from ..utils.prompts import (
    get_acva_prompt,
    get_alghafa_prompt,
    get_boolq_prompt,
    get_ceval_prompt,
    get_sciq_prompt,
)


# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]
# fmt: on

# ARABIC MMLU ##
# fmt: off
ARABIC_MMLU_SUBSETS = [
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


class CustomArabicMMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=get_ceval_prompt("ar"),
            hf_repo="OALL/Arabic_MMLU",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            suite=["custom"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
            trust_dataset=True,
            version=0,
        )


ARABIC_MMLU_TASKS = [
    CustomArabicMMLUTask(name=f"arabic_mmlu:{subset}", hf_subset=subset)
    for subset in ARABIC_MMLU_SUBSETS
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


class CustomACVATask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=get_acva_prompt("ar"),
            hf_repo="OALL/ACVA",
            generation_size=5,
            stop_sequence=["\n"],
            metric=(
                get_qa_metric("ar", "exact"),
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["custom"],
            output_regex=None,
            frozen=False,
            trust_dataset=True,
            version=0,
        )


ACVA_TASKS = [
    CustomACVATask(name=f"acva:{subset}", hf_subset=subset) for subset in ACVA_SUBSETS
]


# ARABIC EXAMS ##

arabic_exams_task = LightevalTaskConfig(
    name="arabic_exams",
    prompt_function=get_ceval_prompt("ar"),
    suite=["custom"],
    hf_repo="OALL/Arabic_EXAMS",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
    ),
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


class CustomAlGhafaNativeTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=get_alghafa_prompt("ar"),
            hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["custom"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
            version=0,
        )


ALGHAFA_TASKS = [
    CustomAlGhafaNativeTask(name=f"alghafa:{subset}", hf_subset=subset)
    for subset in ALGHAFA_SUBSETS
]


# ALGHAFA TRANSLATED ##
# race_ar
race_ar_task = LightevalTaskConfig(
    name="race_ar",
    prompt_function=get_alghafa_prompt("ar"),
    suite=["custom"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="race_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
    ),
    trust_dataset=True,
    version=0,
)


# piqa_ar
piqa_ar_task = LightevalTaskConfig(
    name="piqa_ar",
    prompt_function=get_alghafa_prompt("ar"),
    suite=["custom"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="piqa_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
    ),
    trust_dataset=True,
    version=0,
)


# arc_easy_ar
arc_easy_ar_task = LightevalTaskConfig(
    name="arc_easy_ar",
    prompt_function=get_alghafa_prompt("ar"),
    suite=["custom"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="arc_easy_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
    ),
    trust_dataset=True,
    version=0,
)


# openbook_qa_ext_ar
openbook_qa_ext_ar_task = LightevalTaskConfig(
    name="openbook_qa_ext_ar",
    prompt_function=get_alghafa_prompt("ar"),
    suite=["custom"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="openbook_qa_ext_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
    ),
    trust_dataset=True,
    version=0,
)


# boolq_ar
boolq_ar_task = LightevalTaskConfig(
    name="boolq_ar",
    prompt_function=get_boolq_prompt("ar"),
    suite=["custom"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="boolq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    generation_size=5,
    stop_sequence=["\n"],
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
        get_qa_metric("ar", "exact"),
    ),
    trust_dataset=True,
    version=0,
)


# NO copa as it's in xcopa
# NO Hellaswag/MMLU/Arc since they are already in MLMM


# toxigen_ar
# TODO: I Don't think this will work so will not convert it


def toxigen_prompt_arabic(line, task_name: str = None):
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
        target_for_fewshot_sorting="نعم" if label == 1 else "لا",
        uncoditioned_prefix="",
    )


toxigen_ar_task = LightevalTaskConfig(
    name="toxigen_ar",
    prompt_function=toxigen_prompt_arabic,
    suite=["custom"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="toxigen_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
    ),
    trust_dataset=True,
    version=0,
)

sciq_ar_task = LightevalTaskConfig(
    name="sciq_ar",
    prompt_function=get_sciq_prompt("ar"),
    suite=["custom"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="sciq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=(
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
        Metrics.loglikelihood_acc_norm_pmi,
    ),
    trust_dataset=True,
    version=0,
)


MC_TASKS = [
    *ARABIC_MMLU_TASKS,
    *ACVA_TASKS,
    *ALGHAFA_TASKS,
    arabic_exams_task,
    race_ar_task,
    piqa_ar_task,
    arc_easy_ar_task,
    openbook_qa_ext_ar_task,
    toxigen_ar_task,
    sciq_ar_task,
]

GENERATIVE_TASKS = [
    *ACVA_TASKS,
    boolq_ar_task
]
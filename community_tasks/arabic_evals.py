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

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import random
import re

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES


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
            prompt_function="mmlu_arabic",
            hf_repo="OALL/Arabic_MMLU",
            metric=["loglikelihood_acc_norm"],
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
            trust_dataset=True,
        )


ARABIC_MMLU_TASKS = [
    CustomArabicMMLUTask(name=f"arabic_mmlu:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_SUBSETS
]


def mmlu_arabic(line, task_name: str = None):
    topic = line["subject"]
    instruction = f"الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول {topic.replace('_', ' ')}. \n\n"
    choices = [line["A"], line["B"], line["C"], line["D"]]
    # Answers are provided with roman letters - we look for the correct index in LETTER_INDICES,
    # it will then be applied to arabic letters
    gold_ix = LETTER_INDICES.index(line["answer"])

    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES_AR[:4], choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=gold_ix,
        instruction=instruction,
        target_for_fewshot_sorting=LETTER_INDICES_AR[gold_ix],
    )


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
            prompt_function="acva",
            hf_repo="OALL/ACVA",
            metric=["loglikelihood_acc_norm"],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
            trust_dataset=True,
        )


ACVA_TASKS = [CustomACVATask(name=f"acva:{subset}", hf_subset=subset) for subset in ACVA_SUBSETS]


def acva(line, task_name: str = None):
    question = line["question"]
    answer = line["answer"]

    return Doc(
        task_name=task_name,
        query=f"السؤال: {question}\nالإجابة:",
        choices=["صح", "خطأ"],
        gold_index=["صح", "خطأ"].index(answer),
    )


# ARABIC EXAMS ##
arabic_exams_task = LightevalTaskConfig(
    name="arabic_exams",
    prompt_function="arabic_exams",
    suite=["community"],
    hf_repo="OALL/Arabic_EXAMS",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


def arabic_exams(line, task_name: str = None):
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
        target_for_fewshot_sorting=choices[answer_index],
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
            prompt_function="alghafa_prompt",
            hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            metric=["loglikelihood_acc_norm"],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
        )


ALGHAFA_TASKS = [CustomAlGhafaNativeTask(name=f"alghafa:{subset}", hf_subset=subset) for subset in ALGHAFA_SUBSETS]


def alghafa_prompt(line, task_name: str = None):
    question = line["query"]
    answer_index = int(line["label"])
    # Dynamically determining the choices by excluding '__few_shots', 'query' and 'label'
    choices_keys = [key for key in line.keys() if key not in ["query", "label", "__few_shots"]]
    choices = [line[key] for key in choices_keys]

    instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
    query = f"{instruction}السؤال: {question}\n"
    for index, choice in enumerate(choices):
        query += f"{index}) {choice}\n"
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
        target_for_fewshot_sorting=choices[answer_index],
    )


# ALGHAFA TRANSLATED ##
# race_ar
race_ar_task = LightevalTaskConfig(
    name="race_ar",
    prompt_function="alghafa_prompt",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="race_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


# piqa_ar
piqa_ar_task = LightevalTaskConfig(
    name="piqa_ar",
    prompt_function="alghafa_prompt",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="piqa_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


# arc_easy_ar
arc_easy_ar_task = LightevalTaskConfig(
    name="arc_easy_ar",
    prompt_function="alghafa_prompt",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="arc_easy_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


# arc_challenge_okapi_ar
arc_challenge_okapi_ar_task = LightevalTaskConfig(
    name="arc_challenge_okapi_ar",
    prompt_function="alghafa_prompt",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="arc_challenge_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


# mmlu_okapi_ar
mmlu_okapi_ar_task = LightevalTaskConfig(
    name="mmlu_okapi_ar",
    prompt_function="alghafa_prompt",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="mmlu_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


# openbook_qa_ext_ar
openbook_qa_ext_ar_task = LightevalTaskConfig(
    name="openbook_qa_ext_ar",
    prompt_function="alghafa_prompt",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="openbook_qa_ext_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


# boolq_ar
boolq_ar_task = LightevalTaskConfig(
    name="boolq_ar",
    prompt_function="boolq_prompt_arabic",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="boolq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


def boolq_prompt_arabic(line, task_name: str = None):
    question = line["question"]
    passage = line["passage"]
    answer = "نعم" if line["answer"] else "لا"
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
        target_for_fewshot_sorting=answer,
    )


# copa_ext_ar
copa_ext_ar_task = LightevalTaskConfig(
    name="copa_ext_ar",
    prompt_function="copa_prompt_arabic",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="copa_ext_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


def copa_prompt_arabic(line, task_name: str = None):
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
        target_for_fewshot_sorting=choices[answer],
    )


# hellaswag_okapi_ar
hellaswag_okapi_ar_task = LightevalTaskConfig(
    name="hellaswag_okapi_ar",
    prompt_function="hellaswag_prompt_arabic",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="hellaswag_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


def hellaswag_prompt_arabic(line, task_name: str = None):
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
        target_for_fewshot_sorting=endings[answer_index],
    )


# toxigen_ar
toxigen_ar_task = LightevalTaskConfig(
    name="toxigen_ar",
    prompt_function="toxigen_prompt_arabic",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="toxigen_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


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
    )


# sciq_ar
sciq_ar_task = LightevalTaskConfig(
    name="sciq_ar",
    prompt_function="sciq_prompt_arabic",
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="sciq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=["loglikelihood_acc_norm"],
    trust_dataset=True,
)


def sciq_prompt_arabic(line, task_name: str = None):
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
        target_for_fewshot_sorting=choices[answer_index],
    )


_TASKS = (
    ARABIC_MMLU_TASKS
    + ACVA_TASKS
    + ALGHAFA_TASKS
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
)

# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))

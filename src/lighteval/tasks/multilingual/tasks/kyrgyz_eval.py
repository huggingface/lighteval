"""
name:
Kyrgyz Evals

dataset:
TTimur/kyrgyzMMLU, TTimur/kyrgyzRC, TTimur/hellaswag_kg,
TTimur/winogrande_kg, TTimur/truthfulqa_kg, TTimur/gsm8k_kg, TTimur/boolq_kg

abstract:
Comprehensive evaluation suite for Kyrgyz language understanding, including MMLU,
Reading Comprehension, HellaSwag, Winogrande, TruthfulQA, GSM8K, and BoolQ tasks.

languages:
kyrgyz

tags:
knowledge, reading-comprehension, common-sense, reasoning, math, truthfulness

paper:
https://ieeexplore.ieee.org/document/11206960
"""

import json
import string
from functools import partial

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LETTER_INDICES = string.ascii_uppercase

# ============================================
# ====== MMLU TASKS ==========================
# ============================================

MMLU_SUBSETS = [
    "kyrgyz_mmlu_all",
    "kyrgyz_mmlu_history",
    "kyrgyz_mmlu_literature",
    "kyrgyz_mmlu_medicine",
    "kyrgyz_mmlu_lang",
    "kyrgyz_mmlu_biology",
    "kyrgyz_mmlu_chemistry",
    "kyrgyz_mmlu_geography",
    "kyrgyz_mmlu_math",
    "kyrgyz_mmlu_physics",
]


def kyrgyz_mmlu_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Creates a prompt for MMLU-style multiple-choice tasks in Kyrgyz.
    """
    question = line["Суроо (KG)"]
    correct_answer = str(line["Туура жооп"])

    choices = [line["А (KG)"], line["Б (KG)"], line["В (KG)"], line["Г (KG)"], line["Д (KG)"]]
    choices = [c.strip() for c in choices if c]

    letter_to_index = {"а": 0, "б": 1, "в": 2, "г": 3, "д": 4}
    gold_index = letter_to_index.get(correct_answer.lower(), 0)

    instruction = "Сиз билимиңизге жана жөндөмүңүзгө жараша суроолорго жооп берген AIсыз. Сизге суроо жана 2-5 жооп варианты берилет, туура жооптун НОМЕРИН (индексин) гана кайтарышыңыз керек.\n\n"

    query = f"{instruction}Суроо: {question}\n\nСунушталган жооптор:\n"

    for i, choice in enumerate(choices):
        if choice:
            query += f"{i}. {choice}\n"

    query += "\n\nТуура жоопту тандаңыз:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
    )


class CustomKyrgyzMMLUTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=partial(kyrgyz_mmlu_prompt),
            hf_repo="TTimur/kyrgyzMMLU",
            metrics=[
                Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()}),
                Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbTokenNorm()}),
            ],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


MMLU_TASKS = [CustomKyrgyzMMLUTask(name=f"kyrgyz_evals:{subset}", hf_subset=subset) for subset in MMLU_SUBSETS]


# ============================================
# ====== READING COMPREHENSION TASKS =========
# ============================================

RC_SUBSETS = [
    "kyrgyz_rc_all",
    "kyrgyz_rc_literature",
    "kyrgyz_rc_math",
    "kyrgyz_rc_news",
    "kyrgyz_rc_wiki",
]


def kyrgyz_rc_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Creates a prompt for Reading Comprehension tasks in Kyrgyz.
    """
    text = line["Текст (KG)"]
    question = line["Суроо (KG)"]
    correct_answer = str(line["Туура жооп"])

    choices = [line["А (KG)"], line["Б (KG)"], line["В (KG)"], line["Г (KG)"]]
    choices = [c.strip() for c in choices if c]

    letter_to_index = {
        "а": 0,
        "б": 1,
        "в": 2,
        "г": 3,
    }
    gold_index = letter_to_index.get(correct_answer.lower(), 0)

    instruction = "Сизге бир темага байланыштуу бир нече үзүндү текст берилген. Бардык үзүндүлөрдү кунт коюп окуп, андан кийин төмөндөгү суроолорго жооп бериңиздер. Суроо менен 2-4 жооп варианты берилет, туура жооптун НОМЕРИН (индексин) гана кайтарышыңыз керек.\n\n"

    query = f"{instruction}Текст: {text}\n\nСуроо: {question}\n\nСунушталган жооптор:\n"

    for i, choice in enumerate(choices):
        if choice:
            query += f"{i}. {choice}\n"

    query += "\n\nТуура жоопту тандаңыз:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
    )


class CustomKyrgyzRCTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=partial(kyrgyz_rc_prompt),
            hf_repo="TTimur/kyrgyzRC",
            metrics=[
                Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()}),
                # Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbTokenNorm()}),
            ],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=-1,
            stop_sequence=None,
            # trust_dataset=True,
            version=0,
        )


RC_TASKS = [CustomKyrgyzRCTask(name=f"kyrgyz_evals:{subset}", hf_subset=subset) for subset in RC_SUBSETS]


# ============================================
# ====== HELLASWAG TASK ======================
# ============================================


def kyrgyz_hellaswag_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Hellaswag-style multiple-choice prompt.

    The Kyrgyz dataset provides:
      - ctx_a_kg, ctx_b_kg: context pieces
      - activity_label_kg: short description
      - endings_kg: list of 4 full candidate endings (strings)
      - label: correct ending index in [0, 3]
    """
    import ast

    ctx_a_kg = line["ctx_a_kg"] if line["ctx_a_kg"] else "."
    ctx_b_kg = line["ctx_b_kg"].capitalize() if line["ctx_b_kg"] else "."

    endings_kg = line.get("endings_kg")

    if isinstance(endings_kg, str):
        try:
            # Try JSON first
            endings_kg = json.loads(endings_kg)
        except Exception:
            try:
                endings_kg = ast.literal_eval(endings_kg)
            except Exception:
                endings_kg = [endings_kg]

    if not isinstance(endings_kg, list):
        endings_kg = [str(endings_kg)]

    endings_kg = endings_kg[:4]

    query = (
        "Төмөндө жалпы түшүнүккө (common sense) байланыштуу бир нече тандоо суроолору (жооптору менен) берилген.\n\n"
    )
    query += f"Суроо: {line['activity_label_kg']}: {ctx_a_kg} {ctx_b_kg}\n"
    query += "".join([f"{letter}. {choice}\n" for letter, choice in zip(LETTER_INDICES, endings_kg)])
    query += "Туура жоопту тандаңыз: "

    gold_ix = int(line["label"]) if line.get("label", "") != "" else -1

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {letter}" for letter in LETTER_INDICES[: len(endings_kg)]],
        gold_index=gold_ix,
        instruction="Төмөндө жалпы түшүнүккө (common sense) байланыштуу бир нече тандоо суроолору (жооптору менен) берилген.\n\n",
    )


HELLASWAG_TASK = LightevalTaskConfig(
    name="kyrgyz_evals:hellaswag_kg",
    prompt_function=kyrgyz_hellaswag_prompt,
    hf_repo="TTimur/hellaswag_kg",
    hf_subset="default",
    metrics=[
        Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()}),
        Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbTokenNorm()}),
    ],
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="sequential",
    generation_size=-1,
    # trust_dataset=True,
    version=0,
)


# ============================================
# ====== WINOGRANDE TASK =====================
# ============================================


def kyrgyz_winogrande_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Creates a prompt for Winogrande tasks in Kyrgyz.
    """
    query, end_of_target = line["sentence_kg"].split("_")
    end_of_target = end_of_target.strip()

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1_kg']} {end_of_target}", f"{line['option2_kg']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,
    )


WINOGRANDE_TASK = LightevalTaskConfig(
    name="kyrgyz_evals:winogrande_kg",
    prompt_function=kyrgyz_winogrande_prompt,
    hf_repo="TTimur/winogrande_kg",
    hf_subset="default",
    metrics=[
        Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()}),
    ],
    hf_avail_splits=["train", "dev"],
    evaluation_splits=["dev"],
    few_shots_split="train",
    few_shots_select="sequential",
    generation_size=-1,
    version=0,
)


# ============================================
# ====== TRUTHFULQA TASK =====================
# ============================================


def kyrgyz_truthful_qa_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Creates a prompt for TruthfulQA tasks in Kyrgyz.
    """
    import ast

    mc1 = line.get("mc1_targets_kg", "{}")
    mc2 = line.get("mc2_targets_kg", "{}")

    if isinstance(mc1, str):
        try:
            mc1 = ast.literal_eval(mc1)
        except (ValueError, SyntaxError):
            mc1 = {"choices": [], "labels": []}
    else:
        mc1 = {"choices": [], "labels": []}

    if isinstance(mc2, str):
        try:
            mc2 = ast.literal_eval(mc2)
        except (ValueError, SyntaxError):
            mc2 = {"choices": [], "labels": []}
    else:
        mc2 = {"choices": [], "labels": []}

    choices = [f" {c}" for c in mc1.get("choices", [])] + [f" {c}" for c in mc2.get("choices", [])]
    labels = mc1.get("labels", []) + mc2.get("labels", [])

    return Doc(
        task_name=task_name,
        query=f"Суроо: {line['Question_kg']}\nЖооп:",
        choices=choices,
        gold_index=[ix for ix, label in enumerate(labels) if label == 1],
        specific={"len_mc1": len(mc1.get("choices", []))},
    )


TRUTHFULQA_TASK = LightevalTaskConfig(
    name="kyrgyz_evals:truthfulqa_mc_kg",
    prompt_function=kyrgyz_truthful_qa_prompt,
    hf_repo="TTimur/truthfulqa_kg",
    hf_subset="default",
    metrics=[Metrics.truthfulqa_mc_metrics],
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    generation_size=-1,
    # trust_dataset=True,
    version=0,
)


# ============================================
# ====== GSM8K TASK ==========================
# ============================================


def kyrgyz_gsm8k_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Creates a prompt for GSM8K tasks in Kyrgyz.
    """
    return Doc(
        task_name=task_name,
        query=f"Суроо: {line['question_kg']}\nЖооп:",
        choices=[f" {line['answer_kg']}"],
        gold_index=0,
    )


GSM8K_TASK = LightevalTaskConfig(
    name="kyrgyz_evals:gsm8k_kg",
    prompt_function=kyrgyz_gsm8k_prompt,
    hf_repo="TTimur/gsm8k_kg",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metrics=[
        Metrics.expr_gold_metric,
        # MultilingualQuasiExactMatchMetric(Language.KIRGHIZ, "full"),
    ],
    stop_sequence=["Суроо:"],
    version=0,
)


# ============================================
# ====== BOOLQ TASK ==========================
# ============================================


def kyrgyz_boolq_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Creates a prompt for BoolQ tasks in Kyrgyz.
    """
    question = line["question_kg"][:-1] if line["question_kg"][-2:] == "??" else line["question_kg"]

    return Doc(
        task_name=task_name,
        query=f"Текст: {line['passage_kg']}\nСуроо: {question}\nЖооп:",
        choices=[" ооба", " жок"],
        gold_index=["ооба", "жок"].index(line["answer_kg"]),
    )


BOOLQ_TASK = LightevalTaskConfig(
    name="kyrgyz_evals:boolq_kg",
    prompt_function=kyrgyz_boolq_prompt,
    hf_repo="TTimur/boolq_kg",
    hf_subset="default",
    metrics=[
        # Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()}),
    ],
    hf_avail_splits=["train", "val"],
    evaluation_splits=["val"],
    few_shots_split="train",
    few_shots_select="sequential",
    generation_size=-1,
    version=0,
)


# ============================================
# ====== TASKS TABLE =========================
# ============================================

TASKS_TABLE = (
    MMLU_TASKS
    + RC_TASKS
    + [
        HELLASWAG_TASK,
        WINOGRANDE_TASK,
        TRUTHFULQA_TASK,
        GSM8K_TASK,
        BOOLQ_TASK,
    ]
)

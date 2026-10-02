"""
name:
emirati Evals

dataset:
tiiuae/alyah-emirati-benchmark

abstract:
Collection of benchmarks for Arabic language.

languages:
arabic

tags:
knowledge, multilingual, multiple-choice

paper:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LETTER_INDICES_AR = [
    "أ",
    "ب",
    "ج",
    "د",
    "هـ",
    "و",
    "ز",
    "ح",
    "ط",
    "ي",
    "ك",
    "ل",
    "م",
    "ن",
    "س",
    "ع",
    "ف",
    "ص",
    "ق",
    "ر",
    "ش",
    "ت",
    "ث",
    "خ",
    "ذ",
    "ض",
    "ظ",
    "غ",
]


def emirati_syn_bench_pfn(line, task_name: str = None):
    question = line["query"]

    # correct_answer is 1–4 → convert to 0–3 for eval
    answer_index = int(line["correct_answer"]) - 1

    allowed_keys = [f"option_{i}" for i in range(1, 5)]
    extracted_choices = [line[key] for key in allowed_keys if key in line]

    instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
    query = f"{instruction}السؤال: {question}\n"

    for index, choice in enumerate(extracted_choices):
        query += f"{LETTER_INDICES_AR[index]}) {choice}\n"

    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=answer_index,
        instruction=instruction,
    )


alyah_task = LightevalTaskConfig(
    name="alyah",
    prompt_function=emirati_syn_bench_pfn,
    hf_repo="tiiuae/alyah-emirati-benchmark",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


TASKS_TABLE = [alyah_task]

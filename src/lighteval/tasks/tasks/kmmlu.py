"""
name:
KMMLU

dataset:
HAERAE-HUB/KMMLU

abstract:
KMMLU is 35,030 expert-level multiple-choice questions in Korean across 45
subjects, collected from original Korean exams (not translated from English).
This port mirrors the `kmmlu` (default, log-likelihood) task of EleutherAI
lm-evaluation-harness v0.4.13 so both tools score the same prompt the same way.

languages:
korean

tags:
knowledge, multiple-choice

paper:
https://arxiv.org/abs/2402.11548
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Dataset commit pinned so scores stay comparable (main on 2026-10-08, used for the lm-eval comparison).
KMMLU_REVISION = "d61b3f19e552c576bf5960dd24289763edc36a88"

# The 45 HF configs. Task name is kmmlu:<suffix>; lm-eval names the same subject kmmlu_<suffix>.
KMMLU_SUBSETS = [
    "Accounting",
    "Agricultural-Sciences",
    "Aviation-Engineering-and-Maintenance",
    "Biology",
    "Chemical-Engineering",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Construction",
    "Criminal-Law",
    "Ecology",
    "Economics",
    "Education",
    "Electrical-Engineering",
    "Electronics-Engineering",
    "Energy-Management",
    "Environmental-Science",
    "Fashion",
    "Food-Processing",
    "Gas-Technology-and-Engineering",
    "Geomatics",
    "Health",
    "Industrial-Engineer",
    "Information-Technology",
    "Interior-Architecture-and-Design",
    "Korean-History",
    "Law",
    "Machine-Design-and-Manufacturing",
    "Management",
    "Maritime-Engineering",
    "Marketing",
    "Materials-Engineering",
    "Math",
    "Mechanical-Engineering",
    "Nondestructive-Testing",
    "Patent",
    "Political-Science-and-Sociology",
    "Psychology",
    "Public-Safety",
    "Railway-and-Automotive-Engineering",
    "Real-Estate",
    "Refrigerating-Machinery",
    "Social-Welfare",
    "Taxation",
    "Telecommunications-and-Wireless-Technology",
]


def subset_suffix(subset: str) -> str:
    return subset.lower().replace("-", "_")


def kmmlu_prompt(line, task_name: str = None):
    # Same string as lm-eval's kmmlu/default/_default_kmmlu_yaml doc_to_text; the colon is full-width (U+FF1A).
    query = f"{line['question'].strip()}\nA. {line['A']}\nB. {line['B']}\nC. {line['C']}\nD. {line['D']}\n정답："
    return Doc(
        task_name=task_name,
        query=query,
        # lm-eval joins context and choice with target_delimiter=" ".
        choices=[" A", " B", " C", " D"],
        gold_index=line["answer"] - 1,  # dataset answer is 1-based
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"kmmlu:{subset_suffix(subset)}",
        prompt_function=kmmlu_prompt,
        hf_repo="HAERAE-HUB/KMMLU",
        hf_subset=subset,
        hf_revision=KMMLU_REVISION,
        hf_avail_splits=["train", "dev", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        metrics=[Metrics.loglikelihood_acc],
        version=0,
    )
    for subset in KMMLU_SUBSETS
]

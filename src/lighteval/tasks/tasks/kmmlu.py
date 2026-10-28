"""
name:
KMMLU

dataset:
HAERAE-HUB/KMMLU

abstract:
KMMLU is 35,030 expert-level multiple-choice questions in Korean across 45
subjects, collected from original Korean exams (not translated from English).
This port includes the `kmmlu` (default, log-likelihood) and `kmmlu_direct`
(generative, standard five-shot with few-shot seed 1234) tasks of EleutherAI
lm-evaluation-harness v0.4.13.

languages:
korean

tags:
knowledge, multiple-choice

paper:
https://arxiv.org/abs/2402.11548
"""

import random
import string
from functools import lru_cache

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
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


@lru_cache(maxsize=None)
def _kmmlu_direct_fewshot_order(index: int) -> tuple[int, ...]:
    # lm-eval samples from all five dev examples once per test row, using seed
    # 1234. Replay by original row index because lighteval shuffles test rows.
    rng = random.Random(1234)
    for _ in range(index + 1):
        order = rng.sample(range(5), 5)
    return tuple(order)


class KMMLUDirectDoc(Doc):
    """Keep the standard five-shot context independent of evaluation ordering."""

    def __setattr__(self, name, value):
        if name == "fewshot_samples" and value:
            if len(value) != 5:
                raise ValueError("kmmlu_direct supports zero-shot or the standard five-shot evaluation.")
            # Use IDs to restore dataset order even when reassigning an already
            # ordered context. Only direct-task documents use this behavior.
            pool = sorted(value, key=lambda doc: int(doc.id))
            value = [pool[i] for i in _kmmlu_direct_fewshot_order(int(self.id))]
        super().__setattr__(name, value)


def kmmlu_direct_prompt(line, task_name: str = None):
    return KMMLUDirectDoc(
        task_name=task_name,
        query=kmmlu_prompt(line, task_name).query,
        choices=["A", "B", "C", "D"],
        gold_index=line["answer"] - 1,
    )


def kmmlu_direct_normalize(text: str) -> str:
    return text.replace(" ", "").lower().translate(str.maketrans("", "", string.punctuation))


kmmlu_direct_exact_match = SampleLevelMetric(
    metric_name="exact_match",
    sample_level_fn=ExactMatches(
        normalize_gold=kmmlu_direct_normalize,
        normalize_pred=kmmlu_direct_normalize,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


KMMLU_TASKS = [
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

KMMLU_DIRECT_TASKS = [
    LightevalTaskConfig(
        name=f"kmmlu_direct:{subset_suffix(subset)}",
        prompt_function=kmmlu_direct_prompt,
        hf_repo="HAERAE-HUB/KMMLU",
        hf_subset=subset,
        hf_revision=KMMLU_REVISION,
        hf_avail_splits=["train", "dev", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        metrics=[kmmlu_direct_exact_match],
        stop_sequence=["Q:", "\n\n", "</s>", "."],
        generation_size=256,
        version=0,
    )
    for subset in KMMLU_SUBSETS
]

TASKS_TABLE = KMMLU_TASKS + KMMLU_DIRECT_TASKS

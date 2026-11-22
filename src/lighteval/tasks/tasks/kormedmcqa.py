"""
name:
KorMedMCQA

dataset:
sean0042/KorMedMCQA

abstract:
KorMedMCQA is a Korean multiple-choice question answering benchmark built from
the Korean healthcare professional licensing examinations (doctor, nurse,
pharmacist and dentist) from 2012 to 2024. Each question has five options. The
test split has 3,009 questions across the four subsets.

languages:
korean

tags:
medical, multiple-choice, knowledge

paper:
https://arxiv.org/abs/2403.01469
"""

import re

import numpy as np

from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


LETTERS = ["A", "B", "C", "D", "E"]


def kormedmcqa_prompt(line, task_name: str = None):
    """Build the lm-evaluation-harness prompt: question, options A-E, then "정답：" (full-width colon)."""
    query = f"{line['question'].strip()}\n" + "".join(f"{letter}. {line[letter]}\n" for letter in LETTERS) + "정답："
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(LETTERS),
        gold_index=line["answer"] - 1,
    )


# Answer extraction regexes from lm-evaluation-harness, kept unchanged for parity:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kormedmcqa/utils.py
CHOICE_PATTERNS = [
    re.compile(r"정답[:\s]*([ABCDE])", re.IGNORECASE),
    re.compile(r"정답은\s*([ABCDE])\s*입니다", re.IGNORECASE),
    re.compile(r"\b([ABCDE])\.", re.IGNORECASE),
    re.compile(r"\b([ABCDE])\b", re.IGNORECASE),
]


def extract_choice(output: str) -> str:
    """Return the last match of the first matching pattern, upper-cased, or "" if none match."""
    for pattern in CHOICE_PATTERNS:
        matches = pattern.findall(output)
        if matches:
            return matches[-1].upper()
    return ""


kormedmcqa_metric = SampleLevelMetric(
    metric_name="em",
    sample_level_fn=ExactMatches(strip_strings=True, normalize_pred=extract_choice),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

SUBSETS = ["doctor", "nurse", "pharm", "dentist"]

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"kormedmcqa:{subset}",
        prompt_function=kormedmcqa_prompt,
        hf_repo="sean0042/KorMedMCQA",
        hf_subset=subset,
        hf_revision="79efd6f91edfc8036330d7a4daa88b9f2deb9a82",
        hf_avail_splits=["train", "dev", "test", "fewshot"],
        evaluation_splits=["test"],
        # Same fixed five few-shot examples for every question, as in the harness.
        few_shots_split="fewshot",
        few_shots_select="sequential",
        generation_size=1024,
        metrics=[kormedmcqa_metric],
        stop_sequence=["Q:", "</s>", "<|im_end|>", ".", "\n\n"],
    )
    for subset in SUBSETS
]

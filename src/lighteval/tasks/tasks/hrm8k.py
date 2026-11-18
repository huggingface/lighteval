"""
name:
HRM8K

dataset:
HAERAE-HUB/HRM8K

abstract:
HRM8K is a Korean math reasoning benchmark with 8,011 problems in 5 subsets.
GSM8K, MATH, OMNI_MATH and MMMLU are Korean translations of English benchmarks,
and KSM comes from original Korean exams and olympiads. The prompt and generation
settings follow the lm-evaluation-harness `hrm8k` task. The data is pinned and kept
as is. In KSM, rows 3, 109, 201 and 1427 (0-indexed) have wrong or malformed golds,
and rows 1254 and 1262 look wrong.

languages:
korean

tags:
math, reasoning

paper:
https://arxiv.org/abs/2501.02448
"""

import numpy as np

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig, LatexExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


INSTRUCTION = "주어진 문제를 풀어보세요.\n문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
MMMLU_INSTRUCTION = (
    "주어진 문제를 풀어보세요.\n문제를 푼 후, 주어진 선택지 (1, 2, 3, 4) 중 최종 선택지를 "
    "다음 형식으로 작성하세요: $\\boxed{N}$.\n\n"
)


def _gold(answer) -> str:
    # `$...$` lets the metric parse KSM golds such as `-8\pi` as LaTeX. Float answers are all integers.
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    return f"${answer}$"


def _doc(instruction: str, line, task_name: str = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=f"{instruction}문제: {line['question'].strip()}\n답변:",
        choices=[_gold(line["answer"])],
        gold_index=0,
    )


def hrm8k_prompt(line, task_name: str = None):
    return _doc(INSTRUCTION, line, task_name)


def hrm8k_mmmlu_prompt(line, task_name: str = None):
    return _doc(MMMLU_INSTRUCTION, line, task_name)


hrm8k_metric = SampleLevelMetric(
    metric_name="extractive_match",
    sample_level_fn=MultilingualExtractiveMatchMetric(
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        # Match boxed first before trying other regexes
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        precision=5,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

HRM8K_SUBSETS = ["GSM8K", "MATH", "OMNI_MATH", "MMMLU", "KSM"]

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"hrm8k:{subset.lower()}",
        prompt_function=hrm8k_mmmlu_prompt if subset == "MMMLU" else hrm8k_prompt,
        hf_repo="HAERAE-HUB/HRM8K",
        hf_subset=subset,
        hf_revision="c360cabf8d733a82455565358b3dc965aab9ba8d",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=2048,
        stop_sequence=[],
        metrics=[hrm8k_metric],
        version=0,
    )
    for subset in HRM8K_SUBSETS
]

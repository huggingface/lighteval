"""
name:
Arabic Mmlu

dataset:
MBZUAI/ArabicMMLU

abstract:
Arabic Mmlu multilingual benchmark.

languages:
arabic

tags:
knowledge, multilingual, multiple-choice

paper:
"""

from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation, normalize_subset
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


ARABIC_MMLU_SUBSETS = [
    "Islamic Studies",
    "Islamic Studies (Middle School)",
    "Islamic Studies (Primary School)",
    "Islamic Studies (High School)",
    "Driving Test",
    "Natural Science (Middle School)",
    "Natural Science (Primary School)",
    "History (Middle School)",
    "History (Primary School)",
    "History (High School)",
    "General Knowledge",
    "General Knowledge (Middle School)",
    "General Knowledge (Primary School)",
    "Law (Professional)",
    "Physics (High School)",
    "Social Science (Middle School)",
    "Social Science (Primary School)",
    "Management (University)",
    "Arabic Language (Middle School)",
    "Arabic Language (Primary School)",
    "Arabic Language (High School)",
    "Political Science (University)",
    "Philosophy (High School)",
    "Accounting (University)",
    "Computer Science (Middle School)",
    "Computer Science (Primary School)",
    "Computer Science (High School)",
    "Computer Science (University)",
    "Geography (Middle School)",
    "Geography (Primary School)",
    "Geography (High School)",
    "Math (Primary School)",
    "Biology (High School)",
    "Economics (Middle School)",
    "Economics (High School)",
    "Economics (University)",
    "Arabic Language (General)",
    "Arabic Language (Grammar)",
    "Civics (Middle School)",
    "Civics (High School)",
]


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu_{Language.ARABIC.value}_{formulation.name.lower()}:{normalize_subset(subset)}",
        prompt_function=get_mcq_prompt_function(
            Language.ARABIC,
            lambda line: {
                "context": line["Context"],
                "question": line["Question"],
                "choices": [str(o) for o in [line[f"Option {i}"] for i in range(1, 6)] if o],
                "gold_idx": ascii_uppercase.index(line["Answer Key"]),
            },
            formulation=formulation,
        ),
        hf_repo="MBZUAI/ArabicMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        hf_avail_splits=["dev"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in ARABIC_MMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

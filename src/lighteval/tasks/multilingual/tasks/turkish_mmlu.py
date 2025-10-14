"""
abstract:
Turkish Mmlu multilingual benchmark.

languages:
turkish

tags:
knowledge, multilingual, multiple-choice
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation, normalize_subset
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = []


TURKISH_MMLU_SUBSET = [
    "Biology",
    "Chemistry",
    "Geography",
    "History",
    "Mathematics",
    "Philosophy",
    "Physics",
    "Religion_and_Ethics",
    "Turkish_Language_and_Literature",
]


turkish_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{Language.TURKISH.value}_{formulation.name.lower()}:{normalize_subset(subset)}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="AYueksel/TurkishMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in TURKISH_MMLU_SUBSET
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

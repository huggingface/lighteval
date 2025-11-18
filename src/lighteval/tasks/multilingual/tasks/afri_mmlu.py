"""
name:
Afri Mmlu

dataset:
masakhane/afrimmlu

abstract:
African MMLU: African Massive Multitask Language Understanding

languages:
amharic, ewe, french, hausa, igbo, kinyarwanda, lingala, luganda, oromo, shona,
sotho, swahili, twi, wolof, xhosa, yoruba, zulu

tags:
knowledge, multilingual, multiple-choice

paper:
https://arxiv.org/abs/2406.03368.
"""

from functools import partial
from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


AFRI_MMLU_SUBSETS = [
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_geography",
    "high_school_microeconomics",
    "international_law",
    "global_facts",
]


afri_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"afri_mmlu_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": ascii_uppercase.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="masakhane/afrimmlu",
        # Temporary until the pr is merged
        hf_revision="refs/pr/1",
        hf_subset=language.value,
        hf_filter=partial(lambda subset, line: line["subject"] == subset, subset),
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
    for subset in AFRI_MMLU_SUBSETS
    for language in [
        Language.AMHARIC,
        # Language.EWE,
        Language.FRENCH,
        # Language.HAUSA,
        # Language.IGBO,
        # Language.KINYARWANDA,
        # Language.LINGALA,
        # Language.LUGANDA,
        # Language.OROMO,
        # Language.SHONA,
        # Language.SOTHO,
        Language.SWAHILI,
        # Language.TWI,
        # Language.WOLOF,
        # Language.XHOSA,
        Language.YORUBA,
        # Language.ZULU,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

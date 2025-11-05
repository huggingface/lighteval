"""
name:
Piqa Ar

dataset:
OALL/AlGhafa-Arabic-LLM-Benchmark-Translated

abstract:
PIQA: Physical Interaction Question Answering PIQA is a benchmark for testing
physical commonsense reasoning. This Arabic version is a translation of the
original PIQA dataset, adapted for Arabic language evaluation. It tests the
ability to reason about physical interactions in everyday situations.

languages:
arabic

tags:
multilingual, multiple-choice, qa, reasoning

paper:
https://arxiv.org/abs/1911.11641
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    alghafa_adapter,
)
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"alghafa_piqa_{Language.ARABIC.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_subset="piqa_ar",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

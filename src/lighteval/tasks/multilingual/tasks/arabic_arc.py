"""
name:
Arabic Arc

dataset:
OALL/AlGhafa-Arabic-LLM-Benchmark-Translated

abstract:
Arabic Arc multilingual benchmark.

languages:
arabic

tags:
multilingual, multiple-choice, reasoning

paper:
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
        name=f"alghafa_arc_{Language.ARABIC.value}_{formulation.name.lower()}:easy",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_subset="arc_easy_ar",
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select="sequential",
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

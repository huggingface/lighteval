"""
name:
Enem

dataset:
maritaca-ai/enem

abstract:
ENEM (Exame Nacional do Ensino Médio) is a standardized Brazilian national
secondary education examination. The exam is used both as a university admission
test and as a high school evaluation test.

languages:
portuguese

tags:
knowledge, multilingual, multiple-choice

paper:
https://huggingface.co/datasets/maritaca-ai/enem
"""

from functools import partial

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    enem_adapter,
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
        name=f"enem_{Language.PORTUGUESE.value}_{formulation.name.lower()}:{year}",
        prompt_function=get_mcq_prompt_function(
            Language.PORTUGUESE,
            partial(
                enem_adapter,
                Language.PORTUGUESE,
            ),
            formulation=formulation,
        ),
        hf_repo="maritaca-ai/enem",
        hf_subset=year,
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for year in ["2022", "2023", "2024"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

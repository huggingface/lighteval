"""
name:
Thai Exams

dataset:
scb10x/thai_exam

abstract:
Thai Exams multilingual benchmark.

languages:
thai

tags:
knowledge, multilingual, multiple-choice

paper:
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    thai_exams_adapter,
)
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


THAI_EXAMS_SUBSETS = ["a_level", "ic", "onet", "tgat", "tpat1"]


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"thai_exams_{Language.THAI.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(Language.THAI, thai_exams_adapter, formulation=formulation),
        hf_repo="scb10x/thai_exam",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for subset in THAI_EXAMS_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

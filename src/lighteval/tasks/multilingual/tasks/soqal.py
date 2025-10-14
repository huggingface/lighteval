"""
abstract:
SOQAL: A large-scale Arabic reading comprehension dataset.

languages:
arabic

tags:
multilingual, qa

paper:
https://arxiv.org/abs/1906.05394
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


TASKS_TABLE = []


soqal_tasks = [
    LightevalTaskConfig(
        name=f"soqal_{Language.ARABIC.value}_{formulation.name.lower()}",
        hf_subset="multiple_choice_grounded_statement_soqal_task",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        evaluation_splits=["test"],
        few_shots_split="validation",
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
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

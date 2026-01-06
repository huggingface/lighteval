"""
name:
Openbook Rus

dataset:
ai-forever/MERA

abstract:
The Russian version is part of the MERA (Multilingual Enhanced Russian NLP
Architectures) project.

languages:
russian

tags:
multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2401.04531
"""

from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
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
        name=f"mera_openbookqa_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["question"],
                "choices": [line["inputs"][f"option_{i.lower()}"] for i in ascii_uppercase[:4]],
                "gold_idx": ascii_uppercase.index(line["outputs"]),
            },
            formulation=formulation,
        ),
        hf_repo="ai-forever/MERA",
        hf_subset="ruopenbookqa",
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

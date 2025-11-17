"""
name:
Openbook Es

dataset:
BSC-LT/openbookqa-es

abstract:
Spanish version of OpenBookQA from BSC Language Technology group

languages:
spanish

tags:
multilingual, multiple-choice, reasoning

paper:
https://huggingface.co/datasets/BSC-LT/openbookqa-es
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
        name=f"openbookqa_{Language.SPANISH.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.SPANISH,
            lambda line: {
                "question": line["question_stem"],
                "choices": line["choices"]["text"],
                "gold_idx": ascii_uppercase.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        hf_repo="BSC-LT/openbookqa-es",
        hf_subset="default",
        evaluation_splits=("test",),
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

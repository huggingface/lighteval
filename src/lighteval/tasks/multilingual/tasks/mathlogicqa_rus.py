"""
name:
Mathlogicqa Rus

dataset:
ai-forever/MERA

abstract:
MathLogicQA is a dataset for evaluating mathematical reasoning in language
models. It consists of multiple-choice questions that require logical reasoning
and mathematical problem-solving. This Russian version is part of the MERA
(Multilingual Evaluation of Reasoning Abilities) benchmark.

languages:
russian

tags:
math, multilingual, qa, reasoning

paper:
https://github.com/ai-forever/MERA
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
        name=f"mathlogic_qa_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["text"],
                "choices": [line["inputs"][f"option_{i.lower()}"] for i in ascii_uppercase[:4]],
                "gold_idx": ascii_uppercase.index(line["outputs"]),
            },
            formulation=formulation,
        ),
        hf_repo="ai-forever/MERA",
        hf_subset="mathlogicqa",
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
        CFFormulation(),
        MCFFormulation(),
        HybridFormulation(),
    ]
]

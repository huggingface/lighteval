"""
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

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = []


mathlogicqa_rus_tasks = [
    LightevalTaskConfig(
        name=f"mathlogic_qa_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["text"],
                "choices": [line["inputs"][f"option_{i.lower()}"] for i in LETTER_INDICES[:4]],
                "gold_idx": LETTER_INDICES.index(line["outputs"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
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

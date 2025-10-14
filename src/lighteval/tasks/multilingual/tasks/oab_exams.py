"""
abstract:
OAB Exams: A collection of questions from the Brazilian Bar Association exam The
exam is required for anyone who wants to practice law in Brazil

languages:
portuguese

tags:
knowledge, multilingual, multiple-choice

paper:
https://huggingface.co/datasets/eduagarcia/oab_exams
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


oab_exams_tasks = [
    LightevalTaskConfig(
        name=f"oab_exams_{Language.PORTUGUESE.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.PORTUGUESE,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="eduagarcia/oab_exams",
        hf_subset="default",
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

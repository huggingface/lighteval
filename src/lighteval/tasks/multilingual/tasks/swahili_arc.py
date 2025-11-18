"""
name:
Swahili Arc

dataset:

abstract:
Swahili Arc multilingual benchmark.

languages:
swahili

tags:
multilingual, multiple-choice, reasoning

paper:
"""

from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
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
        name=f"community_arc_{Language.SWAHILI.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.SWAHILI,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else ascii_uppercase.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        hf_repo=f"Mollel/ARC_{subset.capitalize()}_SWH",
        hf_subset="default",
        hf_revision="5347439d3193c8a0dabaab3819914bf076dc94d4"
        if subset == "easy"
        else "dc1df9df632d14c251594d9129fb833d2ca4429c",
        evaluation_splits=("test",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ]
            + ([LogLikelihoodAccMetric(normalization=LogProbPMINorm())] if subset == "challenge" else []),  # type: ignore
        ),
    )
    for subset in ["easy", "challenge"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

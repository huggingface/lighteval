"""
name:
Hindi Arc

dataset:
ai4bharat/ai2_arc-hi

abstract:
Hindi Arc multilingual benchmark.

languages:
hindi

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
        name=f"community_arc_{Language.HINDI.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.HINDI,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else ascii_uppercase.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        hf_repo="ai4bharat/ai2_arc-hi",
        hf_subset=f"ARC-{subset.capitalize()}",
        evaluation_splits=("test",),
        few_shots_split="validation",
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

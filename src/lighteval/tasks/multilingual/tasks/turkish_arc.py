"""
abstract:
Turkish ARC Comes from the Turkish leaderboard

languages:
turkish

tags:
multilingual, multiple-choice, reasoning
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
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


turkish_arc_tasks = [
    LightevalTaskConfig(
        name=f"community_arc_{Language.TURKISH.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="malhajar/arc-tr",
        hf_subset=f"ARC-{subset.capitalize()}",
        evaluation_splits=("test",),
        hf_avail_splits=["train"],
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

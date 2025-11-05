"""
name:
Hellaswag Tha

dataset:
lighteval/hellaswag_thai

abstract:
Hellaswag Thai This is a Thai adaptation of the Hellaswag task. Similar to the
Turkish version, there's no specific paper, but it has been found to be
effective for evaluating Thai language models on commonsense reasoning tasks.

languages:
thai

tags:
multilingual, multiple-choice, reasoning

paper:
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"community_hellaswag_{Language.THAI.value}_{formulation.name.lower()}",
        prompt_function=get_hellaswag_prompt_function(
            language=Language.THAI,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
            wikihow_artifacts=[" [ชื่อ]", " [ส่วนหัว]", " [ขั้นตอน]", " [header]", " [Header]"],
        ),
        hf_repo="lighteval/hellaswag_thai",
        hf_subset="default",
        evaluation_splits=["validation"],
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

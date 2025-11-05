"""
name:
Hellaswag Tur

dataset:
malhajar/hellaswag_tr-v0.2

abstract:
Hellaswag Turkish This is a Turkish adaptation of the Hellaswag task. While
there's no specific paper for this version, it has been found to work well for
evaluating Turkish language models on commonsense reasoning tasks. We don't
handle them in single task as there is quite a lot of differences
(dataset/subset, dot replacement, etc.) which would make it hard to read

languages:
turkish

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
        name=f"community_hellaswag_{Language.TURKISH.value}_{formulation.name.lower()}",
        prompt_function=get_hellaswag_prompt_function(
            language=Language.TURKISH,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
            # https://github.com/malhajar17/lm-evaluation-harness_turkish/blob/main/lm_eval/tasks/hellaswag_tr-v0.2/utils.py
            wikihow_artifacts=[" [title]", " [başlık]", " [adım]", " [header]"],
        ),
        hf_repo="malhajar/hellaswag_tr-v0.2",
        hf_subset="default",
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
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

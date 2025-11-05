"""
name:
Hellaswag Hin

dataset:
ai4bharat/hellaswag-hi

abstract:
Hellaswag Hin multilingual benchmark.

languages:
hindi

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
        name=f"community_hellaswag_{Language.HINDI.value}_{formulation.name.lower()}",
        prompt_function=get_hellaswag_prompt_function(
            language=Language.HINDI,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="ai4bharat/hellaswag-hi",
        hf_filter=lambda line: all(len(choice.strip()) > 0 for choice in line["endings"]),
        hf_subset="hi",
        evaluation_splits=("validation",),
        few_shots_split="validation",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

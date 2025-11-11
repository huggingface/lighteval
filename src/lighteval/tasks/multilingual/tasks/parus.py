"""
name:
Parus

dataset:
ai-forever/MERA

abstract:
PARus: Plausible Alternatives for Russian PARus is the Russian adaptation of the
COPA task, part of the Russian SuperGLUE benchmark. It evaluates common sense
reasoning and causal inference abilities in Russian language models.

languages:
russian

tags:
multilingual

paper:
https://russiansuperglue.com/tasks/task_info/PARus
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.copa import get_copa_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"parus_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_copa_prompt_function(
            language=Language.RUSSIAN,
            adapter=lambda line: {
                "context": line["inputs"]["premise"],
                "cause_effect": line["meta"]["task"],
                "continuations": [line["inputs"]["choice1"], line["inputs"]["choice2"]],
                "gold_idx": int(line["outputs"]) - 1,
            },
            formulation=formulation,
        ),
        hf_repo="ai-forever/MERA",
        hf_subset="parus",
        evaluation_splits=["train"],
        few_shots_split="validation",
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

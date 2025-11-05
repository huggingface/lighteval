"""
name:
Rcb

dataset:
ai-forever/MERA

abstract:
Russian Commitment Bank (RCB) is a large-scale NLI dataset with Russian
sentences, collected from the web and crowdsourcing.

languages:
russian

tags:
classification, multilingual, nli

paper:
https://arxiv.org/abs/2401.04531
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"rcb_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_nli_prompt_function(
            language=Language.RUSSIAN,
            adapter=lambda line: {
                "premise": line["inputs"]["premise"],
                "hypothesis": line["inputs"]["hypothesis"],
                # Since we ignore the neutral label
                "gold_idx": int(line["outputs"]) - 1,
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_repo="ai-forever/MERA",
        hf_subset="rcb",
        # Ignore neutral label
        hf_filter=lambda x: int(x["outputs"] or "0") in [1, 2],
        evaluation_splits=("train",),
        few_shots_split="validation",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=None),
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

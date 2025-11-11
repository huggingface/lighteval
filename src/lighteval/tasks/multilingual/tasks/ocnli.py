"""
name:
Ocnli

dataset:
clue/clue

abstract:
Native Chinese NLI dataset based.

languages:
chinese

tags:
classification, multilingual, nli

paper:
https://arxiv.org/pdf/2010.05444
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
        name=f"ocnli_{Language.CHINESE.value}_{formulation.name.lower()}",
        prompt_function=get_nli_prompt_function(
            language=Language.CHINESE,
            adapter=lambda line: {
                "premise": line["sentence1"],
                "hypothesis": line["sentence2"],
                # Since we ignore the neutral label
                "gold_idx": {1: 0, 2: 1}[line["label"]],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_repo="clue/clue",
        hf_subset="ocnli",
        # Only keep the positive and negative examples
        hf_filter=lambda x: int(x["label"]) in [1, 2],
        evaluation_splits=("validation",),
        few_shots_split="train",
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

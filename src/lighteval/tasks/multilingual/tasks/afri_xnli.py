"""
name:
Afri Xnli

dataset:
masakhane/afrixnli

abstract:
African XNLI: African XNLI

languages:
amharic, ewe, french, hausa, igbo, kinyarwanda, lingala, luganda, oromo, shona,
sotho, swahili, twi, wolof, xhosa, yoruba, zulu

tags:
classification, multilingual, nli

paper:
https://arxiv.org/abs/2406.03368.
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
        name=f"afri_xnli_{language.value}_{formulation.name.lower()}",
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since we ignore the neutral label
                "gold_idx": {0: 0, 2: 1}[line["label"]],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_repo="masakhane/afrixnli",
        hf_subset=language.value,
        hf_filter=lambda x: int(x["label"]) in [0, 2],
        evaluation_splits=("test",),
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
    for language in [
        Language.AMHARIC,
        # Language.EWE,
        Language.FRENCH,
        # Language.HAUSA,
        # Language.IGBO,
        # Language.KINYARWANDA,
        # Language.LINGALA,
        # Language.LUGANDA,
        # Language.OROMO,
        # Language.SHONA,
        # Language.SOTHO,
        Language.SWAHILI,
        # Language.TWI,
        # Language.WOLOF,
        # Language.XHOSA,
        Language.YORUBA,
        # Language.ZULU,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

"""
name:
Xcopa

dataset:

abstract:
COPA (Choice of Plausible Alternatives) tasks involve determining the most
plausible cause or effect for a given premise. These tasks test common sense
reasoning and causal inference abilities. XCOPA: Cross-lingual Choice of
Plausible Alternatives.

languages:
arabic, chinese, estonian, haitian, indonesian, italian, quechua, swahili,
tamil, thai, turkish, vietnamese

tags:
multilingual, multiple-choice, narrative, reasoning

paper:
https://aclanthology.org/2020.emnlp-main.185/
"""

from langcodes import standardize_tag

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
        name=f"xcopa_{language.value}_{formulation.name.lower()}",
        prompt_function=get_copa_prompt_function(
            language,
            adapter=lambda line: {
                "context": line["premise"],
                "cause_effect": line["question"],
                "continuations": [line["choice1"], line["choice2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo=("OALL/AlGhafa-Arabic-LLM-Benchmark-Translated" if language == Language.ARABIC else "xcopa"),
        hf_subset=("copa_ext_ar" if language == Language.ARABIC else standardize_tag(language.value)),
        evaluation_splits=["test"],
        few_shots_split="validation",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for language in [
        Language.ARABIC,
        Language.ESTONIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.SWAHILI,
        Language.TAMIL,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,
        Language.HAITIAN,
        Language.QUECHUA,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

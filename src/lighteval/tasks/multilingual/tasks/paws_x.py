"""
name:
Paws X

dataset:
google-research-datasets/paws-x

abstract:
PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification This
dataset contains paraphrase identification pairs in multiple languages. It's
derived from PAWS (Paraphrase Adversaries from Word Scrambling) and We treat
paraphrase as entailment and non-paraphrase as contradiction

languages:
chinese, english, french, german, japanese, korean, spanish

tags:
classification, multilingual, nli

paper:
https://arxiv.org/abs/1908.11828
"""

from langcodes import standardize_tag

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
        name=f"pawsx_{language.value}_{formulation.name.lower()}",
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["sentence1"],
                "hypothesis": line["sentence2"],
                # Since we ignore the neutral label
                "gold_idx": int(line["label"]),
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_repo="google-research-datasets/paws-x",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
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
    for language in [
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.JAPANESE,
        Language.KOREAN,
        Language.CHINESE,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

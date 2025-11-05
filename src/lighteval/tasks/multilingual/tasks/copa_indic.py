"""
name:
Copa Indic

dataset:
ai4bharat/IndicCOPA

abstract:
IndicCOPA: COPA for Indic Languages Paper: https://arxiv.org/pdf/2212.05409
IndicCOPA extends COPA to 15 Indic languages, providing a valuable resource for
evaluating common sense reasoning in these languages.

languages:
assamese, bengali, gujarati, hindi, kannada, malayalam, marathi, nepali, oriya,
punjabi, sanskrit, sindhi, tamil, telugu, urdu

tags:
multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/pdf/2212.05409
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


# IndicCOPA: COPA for Indic Languages
# Paper: https://arxiv.org/pdf/2212.05409
# IndicCOPA extends COPA to 15 Indic languages, providing a valuable resource for
# evaluating common sense reasoning in these languages.


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"indicxcopa_{language.value}_{formulation.name.lower()}",
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
        hf_repo="ai4bharat/IndicCOPA",
        hf_subset=f"translation-{standardize_tag(language.value)}",
        hf_revision="d356ef19a4eb287e88a51d07a56b73ba88c7f188",
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for language in [
        Language.ASSAMESE,
        Language.BENGALI,
        Language.GUJARATI,
        Language.HINDI,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.NEPALI,
        Language.ORIYA,
        Language.PUNJABI,
        Language.SANSKRIT,
        Language.SINDHI,
        Language.TAMIL,
        Language.TELUGU,
        Language.URDU,
        # Optionally: Maithili, Santali, Sindhi, Konkani
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

"""
name:
Mlmm Hellaswag

dataset:
jon-tow/okapi_hellaswag

abstract:
Hellaswag is a commonsense reasoning task that requires models to complete a
given scenario with the most plausible ending. It tests the model's ability to
understand and reason about everyday situations and human behavior.
MLMM-Hellaswag: Multilingual adaptation of Hellaswag

languages:
arabic, armenian, basque, bengali, catalan, chinese, croatian, danish, dutch,
french, german, gujarati, hindi, hungarian, icelandic, indonesian, italian,
kannada, malayalam, marathi, nepali, norwegian, portuguese, romanian, russian,
serbian, slovak, spanish, swedish, tamil, telugu, ukrainian, vietnamese

tags:
multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2306.07610
"""

from langcodes import standardize_tag

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
        name=f"mlmm_hellaswag_{lang.value}_{formulation.name.lower()}",
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=lambda line: {
                # We don't use activity_label as they are not available
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(lang.value),
        hf_revision="fb6b5f2b48a6348ae2a9467a6714c3a054fa8d85",
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
    for lang in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CATALAN,
        Language.DANISH,
        Language.GERMAN,
        Language.SPANISH,
        Language.BASQUE,
        Language.FRENCH,
        Language.GUJARATI,
        Language.HINDI,
        Language.CROATIAN,
        Language.HUNGARIAN,
        Language.ARMENIAN,
        Language.INDONESIAN,
        Language.ICELANDIC,
        Language.ITALIAN,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.NORWEGIAN,
        Language.NEPALI,
        Language.DUTCH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.SLOVAK,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.TAMIL,
        Language.TELUGU,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

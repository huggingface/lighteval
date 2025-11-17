"""
name:
Xwinograd

dataset:
Muennighoff/xwinograd

abstract:
Xwinograd multilingual benchmark.

languages:
chinese, english, french, japanese, portuguese, russian

tags:
multilingual, multiple-choice, reasoning

paper:
"""

from functools import partial

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    winogrand_adapter,
)
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"xwinograd_{language.value}_{formulation.name.lower()}",
        prompt_function=get_continuation_prompt_function(
            language, partial(winogrand_adapter, language), formulation=formulation
        ),
        hf_repo="Muennighoff/xwinograd",
        hf_subset=standardize_tag(language.value) if language != Language.JAPANESE else "jp",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metrics=[
            LogLikelihoodAccMetric(normalization=None),
            LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        ],
    )
    for language in [
        Language.ENGLISH,
        Language.FRENCH,
        Language.JAPANESE,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.CHINESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

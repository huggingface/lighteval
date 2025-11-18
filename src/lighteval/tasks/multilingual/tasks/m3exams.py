"""
name:
M3Exams

dataset:
chiayewken/m3exam

abstract:
M3Exam: Multitask Multilingual Multimodal Evaluation Benchmark It also contains
a multimodal version but we don't support that Paper:
https://arxiv.org/abs/2306.05179

languages:
afrikaans, chinese, english, italian, javanese, portuguese, swahili, thai,
vietnamese

tags:
knowledge, multilingual, multiple-choice

paper:
https://arxiv.org/abs/2306.05179
"""

from functools import partial

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    get_m3exam_adapter,
)
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"m3exams_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            language,
            partial(get_m3exam_adapter, language),
            formulation=formulation,
        ),
        hf_repo="chiayewken/m3exam",
        hf_subset=LangCodeLanguage(standardize_tag(language.value)).language_name().lower(),
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=-1,
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for language in [
        Language.AFRIKAANS,
        Language.CHINESE,
        Language.ENGLISH,
        Language.ITALIAN,
        Language.JAVANESE,
        Language.PORTUGUESE,
        Language.SWAHILI,
        Language.THAI,
        Language.VIETNAMESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

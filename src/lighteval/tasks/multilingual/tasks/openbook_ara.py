"""
name:
Openbook Ara

dataset:
OALL/AlGhafa-Arabic-LLM-Benchmark-Translated

abstract:
OpenBookQA: A Question-Answering Dataset for Open-Book Exams OpenBookQA is a
question-answering dataset modeled after open-book exams for assessing human
understanding of a subject. It consists of multiple-choice questions that
require combining facts from a given open book with broad common knowledge. The
task tests language models' ability to leverage provided information and apply
common sense reasoning.

languages:
arabic

tags:
multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/abs/1809.02789
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    alghafa_adapter,
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
        name=f"alghafa_openbookqa_{Language.ARABIC.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="openbook_qa_ext_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

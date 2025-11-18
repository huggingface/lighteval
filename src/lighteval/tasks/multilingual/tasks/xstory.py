"""
name:
Xstory

dataset:
juletxara/xstory_cloze

abstract:
Xstory multilingual benchmark.

languages:
arabic, basque, burmese, chinese, hindi, indonesian, russian, spanish, swahili,
telugu

tags:
multilingual, narrative

paper:
"""

from functools import partial

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"xstory_cloze_{lang.value}_{formulation.name.lower()}",
        prompt_function=get_continuation_prompt_function(
            lang,
            partial(
                lambda lang, line: {
                    "context": TRANSLATION_LITERALS[lang].sentence_space.join(
                        [
                            line["input_sentence_1"],
                            line["input_sentence_2"],
                            line["input_sentence_3"],
                            line["input_sentence_4"],
                        ]
                    ),
                    "continuations": [line["sentence_quiz1"], line["sentence_quiz2"]],
                    "gold_idx": int(line["answer_right_ending"]) - 1,  # type: ignore
                },
                lang,
            ),
            formulation=formulation,
        ),
        hf_repo="juletxara/xstory_cloze",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["eval"],
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for lang in [
        Language.RUSSIAN,
        Language.CHINESE,
        Language.SPANISH,
        Language.ARABIC,
        Language.HINDI,
        Language.INDONESIAN,
        Language.TELUGU,
        Language.SWAHILI,
        Language.BASQUE,
        Language.BURMESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

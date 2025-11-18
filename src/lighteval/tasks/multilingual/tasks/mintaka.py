"""
name:
Mintaka

dataset:
AmazonScience/mintaka

abstract:
Mintaka multilingual benchmark.

languages:
arabic, english, french, german, hindi, italian, japanese, portuguese, spanish

tags:
knowledge, multilingual, qa

paper:
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mintaka_{lang.value}",
        prompt_function=get_qa_prompt_function(
            lang,
            lambda line: {
                "question": line["question"],
                "choices": [line["answerText"]],
            },
        ),
        hf_repo="AmazonScience/mintaka",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metrics=[
            MultilingualQuasiExactMatchMetric(lang, "prefix"),
            MultilingualQuasiF1ScoreMetric(lang),
        ],
    )
    for lang in [
        Language.ARABIC,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.HINDI,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.PORTUGUESE,
    ]
]

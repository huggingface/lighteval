"""
name:
Indicqa

dataset:
ai4bharat/IndicQA

abstract:
IndicQA: A reading comprehension dataset for 11 Indian languages.

languages:
assamese, bengali, gujarati, hindi, kannada, malayalam, marathi, oriya, punjabi,
tamil, telugu

tags:
multilingual, qa

paper:
https://arxiv.org/abs/2407.13522
"""

from langcodes import Language as LangCodeLanguage

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"indicqa_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="ai4bharat/IndicQA",
        hf_subset=f"indicqa.{LangCodeLanguage.get(language.value).language}",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        hf_revision="92d96092ae229950973dac3b9998f8b3a8949b0a",
        evaluation_splits=("test",),
        hf_avail_splits=("test",),
        generation_size=400,
        metrics=(
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ),
        stop_sequence=("\n",),
    )
    for language in [
        Language.ASSAMESE,
        Language.BENGALI,
        Language.GUJARATI,
        Language.HINDI,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.ORIYA,
        Language.PUNJABI,
        Language.TAMIL,
        Language.TELUGU,
    ]
]

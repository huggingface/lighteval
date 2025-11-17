"""
name:
Tydiqa

dataset:
google-research-datasets/tydiqa

abstract:
Other QA tasks for RC TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. https://arxiv.org/abs/2003.05002

languages:
arabic, bengali, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai

tags:
multilingual, qa

paper:
https://arxiv.org/abs/2003.05002
"""

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"tydiqa_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="google-research-datasets/tydiqa",
        hf_subset="secondary_task",
        evaluation_splits=("validation",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metrics=(
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ),
    )
    for language in [
        Language.ENGLISH,
        Language.ARABIC,
        Language.BENGALI,
        Language.FINNISH,
        Language.INDONESIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.SWAHILI,
        Language.RUSSIAN,
        Language.TELUGU,
        Language.THAI,
    ]
]

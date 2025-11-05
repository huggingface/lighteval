"""
name:
Xquad

dataset:
google/xquad

abstract:
Reading Comprehension (RC) tasks evaluate a model's ability to understand and
extract information from text passages. These tasks typically involve answering
questions based on given contexts, spanning multiple languages and formats. Add
RC tasks supporting about 130 unique languages/scripts. SQuAD - like XQuAD:
Cross-lingual Question Answering Dataset, extending SQuAD to 11 languages.

languages:
arabic, chinese, english, german, greek, hindi, romanian, russian, spanish,
thai, turkish, vietnamese

tags:
multilingual, qa

paper:
https://arxiv.org/abs/1910.11856
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
        name=f"xquad_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="google/xquad",
        hf_subset=f"xquad.{standardize_tag(language.value)}",
        evaluation_splits=("validation",),
        few_shots_split="validation",
        generation_size=400,
        stop_sequence=("\n",),
        metrics=(
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ),
    )
    for language in [
        Language.ARABIC,
        Language.GERMAN,
        Language.GREEK,
        Language.ENGLISH,
        Language.SPANISH,
        Language.HINDI,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
]

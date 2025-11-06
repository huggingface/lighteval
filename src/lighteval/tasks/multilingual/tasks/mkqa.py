"""
name:
Mkqa

dataset:
apple/mkqa

abstract:
Mkqa multilingual benchmark.

languages:
arabic, chinese, chinese_hong_kong, chinese_traditional, danish, dutch, english,
finnish, french, german, hebrew, hungarian, italian, japanese, khmer, korean,
malay, norwegian, polish, portuguese, russian, spanish, swedish, thai, turkish,
vietnamese

tags:
multilingual, qa

paper:
"""

from functools import partial

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    get_mkqa_adapter,
)
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


MKQA_TASK_TO_ID = {
    "entity": 0,
    "long_answer": 1,
    # "unanswerable": 2,
    "date": 3,
    "number": 4,
    "number_with_unit": 5,
    "short_phrase": 6,
    "binary": 7,
}


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mkqa_{language.value}:{subset}",
        prompt_function=get_qa_prompt_function(language, partial(get_mkqa_adapter, language)),
        hf_repo="apple/mkqa",
        hf_subset="mkqa",
        hf_revision="325131889721ae0ed885b76ecb8011369d75abad",
        hf_filter=partial(
            lambda language, subset, line: line["answers"][
                "zh_cn" if language == Language.CHINESE else standardize_tag(language.value)
            ][0]["type"]
            == MKQA_TASK_TO_ID[subset],
            language,
            subset,
        ),
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        stop_sequence=("\n",),
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ]
        if subset in ["entity", "long_answer", "short_phrase"]
        else [
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
    )
    for subset in MKQA_TASK_TO_ID.keys()
    for language in [
        Language.ARABIC,
        Language.DANISH,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FINNISH,
        Language.FRENCH,
        Language.HEBREW,
        Language.HUNGARIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.KHMER,
        Language.MALAY,
        Language.DUTCH,
        Language.NORWEGIAN,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SWEDISH,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,  # Simplified
        # Language.CHINESE_HONG_KONG,
        # Language.CHINESE_TRADITIONAL,
    ]
]

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
        name=f"mlqa_{lang.value}",
        prompt_function=get_qa_prompt_function(
            lang,
            lambda line: {
                "context": line["context"],
                "question": line["question"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        # --------------------------------------------------------
        # Bypass the deprecated script by loading the parquet directly
        hf_repo="parquet",
        hf_subset="default",
        hf_data_files={
            "test": f"hf://datasets/facebook/mlqa@refs%2Fconvert%2Fparquet/mlqa.{standardize_tag(lang.value)}.{standardize_tag(lang.value)}/test/*.parquet"
        },
        # (Removed hf_revision since the parquet URL handles it)
        # --------------------------------------------------------
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
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
        Language.SPANISH,
        Language.CHINESE,
        Language.HINDI,
        Language.VIETNAMESE,
    ]
]
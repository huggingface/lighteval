"""
name:
Mlqa

dataset:
facebook/mlqa

abstract:
MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating
cross-lingual question answering performance. It consists of QA instances in 7
languages: English, Arabic, German, Spanish, Hindi, Vietnamese, and Chinese. The
dataset is derived from the SQuAD v1.1 dataset, with questions and contexts
translated by professional translators.

languages:
arabic, chinese, german, hindi, spanish, vietnamese

tags:
multilingual, qa

paper:
https://arxiv.org/abs/1910.07475
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
        name=f"mlqa_{lang.value}",
        prompt_function=get_qa_prompt_function(
            lang,
            lambda line: {
                "context": line["context"],
                "question": line["question"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="facebook/mlqa",
        hf_subset=f"mlqa.{standardize_tag(lang.value)}.{standardize_tag(lang.value)}",
        hf_revision="397ed406c1a7902140303e7faf60fff35b58d285",
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

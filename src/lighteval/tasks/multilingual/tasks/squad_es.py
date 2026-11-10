"""
name:
Squad Es

dataset:
ccasimiro/squad_es

abstract:
SQuAD-es: Spanish translation of the Stanford Question Answering Dataset

languages:
spanish

tags:
multilingual, qa

paper:
https://huggingface.co/datasets/ccasimiro/squad_es
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
        # Replace the original hf_repo and hf_subset with this:
        hf_repo="parquet",
        hf_subset="default",
        hf_data_files={
            # squad_es uses "validation" and "train" splits instead of "test"
            "validation": "hf://datasets/ccasimiro/squad_es@refs%2Fconvert%2Fparquet/v2.0.0/validation/*.parquet",
            "train": "hf://datasets/ccasimiro/squad_es@refs%2Fconvert%2Fparquet/v2.0.0/train/*.parquet"
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

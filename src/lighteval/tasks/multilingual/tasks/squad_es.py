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
        name=f"squad_{Language.SPANISH.value}",
        prompt_function=get_qa_prompt_function(
            Language.SPANISH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="ccasimiro/squad_es",
        hf_subset="v2.0.0",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("validation",),
        few_shots_split="train",
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.SPANISH, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.SPANISH),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

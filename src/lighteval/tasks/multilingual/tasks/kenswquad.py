"""
name:
Kenswquad

dataset:
lighteval/KenSwQuAD

abstract:
KenSwQuAD: A question answering dataset for Kenyan Swahili.

languages:
swahili

tags:
multilingual, qa

paper:
https://arxiv.org/abs/2205.02364
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
        name=f"kenswquad_{Language.SWAHILI.value}",
        prompt_function=get_qa_prompt_function(
            Language.SWAHILI,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [line["answer"]],
            },
        ),
        hf_repo="lighteval/KenSwQuAD",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.SWAHILI, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.SWAHILI),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

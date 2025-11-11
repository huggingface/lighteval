"""
name:
Arcd

dataset:
hsseinmz/arcd

abstract:
ARCD: Arabic Reading Comprehension Dataset.

languages:
arabic

tags:
multilingual, multiple-choice, qa, reasoning

paper:
https://arxiv.org/pdf/1906.05394
"""

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


# ARCD: Arabic Reading Comprehension Dataset.
# https://arxiv.org/pdf/1906.05394


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"arcd_{Language.ARABIC.value}",
        prompt_function=get_qa_prompt_function(
            Language.ARABIC,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="hsseinmz/arcd",
        hf_subset="plain_text",
        evaluation_splits=("validation",),
        few_shots_split="train",
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.ARABIC, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.ARABIC),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

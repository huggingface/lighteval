"""
name:
Fquad V2

dataset:
manu/fquad2_test

abstract:
FQuAD v2: French Question Answering Dataset version 2.

languages:
french

tags:
multilingual, qa

paper:
https://arxiv.org/abs/2002.06071
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
        name=f"fquadv2_{Language.FRENCH.value}",
        prompt_function=get_qa_prompt_function(
            Language.FRENCH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="manu/fquad2_test",
        hf_subset="default",
        evaluation_splits=("test_hasAns",),
        few_shots_split="valid_hasAns",
        generation_size=400,
        stop_sequence=("\n",),
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.FRENCH, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.FRENCH),
        ),
    )
]

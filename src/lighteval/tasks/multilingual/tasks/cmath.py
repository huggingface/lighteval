"""
name:
Cmath

dataset:
weitianwen/cmath

abstract:
Cmath multilingual benchmark.

languages:
chinese

tags:
math, multilingual, reasoning

paper:
"""

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"cmath_{Language.CHINESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["question"],
                "choices": [line["golden"]],
            },
        ),
        hf_repo="weitianwen/cmath",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="validation",
        generation_size=25,
        metrics=[
            MultilingualQuasiExactMatchMetric(Language.CHINESE, "full"),
        ],
        stop_sequence=("\n",),
    )
]

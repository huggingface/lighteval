"""
name:
French Boolq

dataset:
manu/french_boolq

abstract:
French Boolq multilingual benchmark.

languages:
french

tags:
classification, multilingual, qa

paper:
"""

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
    MultilingualQuasiExactMatchMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.boolq import get_boolq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"community_boolq_{Language.FRENCH.value}",
        prompt_function=get_boolq_prompt_function(
            Language.FRENCH,
            lambda line: {
                "question": line["question"],
                "answer": line["label"] == 1,
                "context": line["passage"],
            },
            formulation=CFFormulation(),
        ),
        hf_repo="manu/french_boolq",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="valid",
        generation_size=5,
        stop_sequence=["\n"],
        metrics=[MultilingualQuasiExactMatchMetric(Language.FRENCH, "full"), LogLikelihoodAccMetric()],
    )
]

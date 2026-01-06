"""
name:
Hindi Boolq

dataset:
ai4bharat/boolq-hi

abstract:
Hindi Boolq multilingual benchmark.

languages:
gujarati, hindi, malayalam, marathi, tamil

tags:
classification, multilingual, qa

paper:
"""

from langcodes import standardize_tag

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
        name=f"community_boolq_{language.value}",
        prompt_function=get_boolq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "answer": line["answer"],
                "context": line["passage"],
            },
            formulation=CFFormulation(),
        ),
        hf_repo="ai4bharat/boolq-hi",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("validation",),
        few_shots_split="train",
        generation_size=5,
        stop_sequence=["\n"],
        metrics=[MultilingualQuasiExactMatchMetric(language, "full"), LogLikelihoodAccMetric()],
    )
    for language in [
        Language.HINDI,
        Language.GUJARATI,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.TAMIL,
    ]
]

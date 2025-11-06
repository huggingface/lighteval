"""
name:
Afri Mgsm

dataset:
masakhane/afrimgsm

abstract:
African MGSM: MGSM for African Languages

languages:
amharic, ewe, french, hausa, igbo, kinyarwanda, lingala, luganda, oromo, shona,
sotho, swahili, twi, wolof, xhosa, yoruba, zulu

tags:
math, multilingual, reasoning

paper:
https://arxiv.org/abs/2406.03368.
"""

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"afri_mgsm_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                # The cot is available but we have no use:
                # line["answer"]
                "choices": [str(line["answer_number"])],
            },
        ),
        hf_repo="masakhane/afrimgsm",
        hf_subset=language.value,
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=25,
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=("\n",),
    )
    for language in [
        Language.AMHARIC,
        # Language.EWE,
        Language.FRENCH,
        # Language.HAUSA,
        # Language.IGBO,
        # Language.KINYARWANDA,
        # Language.LINGALA,
        # Language.LUGANDA,
        # Language.OROMO,
        # Language.SHONA,
        # Language.SOTHO,
        Language.SWAHILI,
        # Language.TWI,
        # Language.WOLOF,
        # Language.XHOSA,
        Language.YORUBA,
        # Language.ZULU,
    ]
]

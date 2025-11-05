"""
name:
Acva

dataset:
OALL/ACVA

abstract:
Acva multilingual benchmark.

languages:
arabic

tags:
knowledge, multilingual, multiple-choice

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


ACVA_SUBSET = [
    "Algeria",
    "Ancient_Egypt",
    "Arab_Empire",
    "Arabic_Architecture",
    "Arabic_Art",
    "Arabic_Astronomy",
    "Arabic_Calligraphy",
    "Arabic_Ceremony",
    "Arabic_Clothing",
    "Arabic_Culture",
    "Arabic_Food",
    "Arabic_Funeral",
    "Arabic_Geography",
    "Arabic_History",
    "Arabic_Language_Origin",
    "Arabic_Literature",
    "Arabic_Math",
    "Arabic_Medicine",
    "Arabic_Music",
    "Arabic_Ornament",
    "Arabic_Philosophy",
    "Arabic_Physics_and_Chemistry",
    "Arabic_Wedding",
    "Bahrain",
    "Comoros",
    "Egypt_modern",
    "InfluenceFromAncientEgypt",
    "InfluenceFromByzantium",
    "InfluenceFromChina",
    "InfluenceFromGreece",
    "InfluenceFromIslam",
    "InfluenceFromPersia",
    "InfluenceFromRome",
    "Iraq",
    "Islam_Education",
    "Islam_branches_and_schools",
    "Islamic_law_system",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Mauritania",
    "Mesopotamia_civilization",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Somalia",
    "Sudan",
    "Syria",
    "Tunisia",
    "United_Arab_Emirates",
    "Yemen",
    "communication",
    "computer_and_phone",
    "daily_life",
    "entertainment",
]


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"acva_{Language.ARABIC.value}:{subset}",
        prompt_function=get_boolq_prompt_function(
            Language.ARABIC,
            lambda line: {
                "question": line["question"],
                "answer": line["answer"] == "صح",
            },
            formulation=CFFormulation(),
        ),
        hf_repo="OALL/ACVA",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metrics=[MultilingualQuasiExactMatchMetric(Language.ARABIC, "full"), LogLikelihoodAccMetric()],
        generation_size=5,
        stop_sequence=("\n",),
    )
    for subset in ACVA_SUBSET
]

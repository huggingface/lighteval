"""
name:
Openai Mmlu

dataset:
openai/MMMLU

abstract:
Openai Mmlu multilingual benchmark.

languages:
arabic, bengali, chinese, french, german, hindi, indonesian, italian, japanese,
korean, portuguese, spanish, swahili, yoruba

tags:
knowledge, multilingual, multiple-choice

paper:
"""

from functools import partial
from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"openai_mmlu_{language[0].value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language[0],
            lambda line: {
                "question": line["Question"],
                "choices": [line["A"], line["B"], line["C"], line["D"]],
                "gold_idx": ascii_uppercase.index(line["Answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="openai/MMMLU",
        hf_subset=language[1],
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        hf_filter=partial(lambda subset, x: x["Subject"].lower() == subset, subset),
        hf_revision="038c7808122969ead7456361af05cb8f47d247f8",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for language in [
        (Language.ARABIC, "AR_XY"),
        (Language.BENGALI, "BN_BD"),
        (Language.GERMAN, "DE_DE"),
        (Language.SPANISH, "ES_LA"),
        (Language.FRENCH, "FR_FR"),
        (Language.HINDI, "HI_IN"),
        (Language.INDONESIAN, "ID_ID"),
        (Language.ITALIAN, "IT_IT"),
        (Language.JAPANESE, "JA_JP"),
        (Language.KOREAN, "KO_KR"),
        (Language.PORTUGUESE, "PT_BR"),
        (Language.SWAHILI, "SW_KE"),
        (Language.YORUBA, "YO_NG"),
        (Language.CHINESE, "ZH_CN"),
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

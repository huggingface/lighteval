"""
name:
Global Mmlu

dataset:
CohereForAI/Global-MMLU

abstract:
Translated MMLU using both professional and non-professional translators.
Contains tags for cultural sensitivity.

languages:
amharic, arabic, bengali, chinese, czech, dutch, english, french, german,
hebrew, hindi, indonesian, italian, japanese, korean, malay, norwegian, polish,
portuguese, romanian, russian, serbian, spanish, swahili, swedish, tamil,
telugu, thai, turkish, ukrainian, urdu, vietnamese, yoruba, zulu

tags:
knowledge, multilingual, multiple-choice

paper:
https://huggingface.co/papers/2412.03304
"""

from functools import partial
from string import ascii_uppercase

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
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
        name=f"global_mmlu_{sensitivity_label.lower()}_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [line["option_a"], line["option_b"], line["option_c"], line["option_d"]],
                "gold_idx": ascii_uppercase.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=partial(
            lambda subset, sensitivity_label, x: x["subject"].lower() == subset
            and (
                sensitivity_label == "ALL" or sensitivity_label in x["cultural_sensitivity_label"].replace("-", "UNK")
            )
            and all(x[f"option_{opt}"] is not None and x[f"option_{opt}"].strip() for opt in "abcd"),
            subset,
            sensitivity_label,
        ),
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
        Language.AMHARIC,
        Language.ARABIC,
        Language.BENGALI,
        Language.CHINESE,
        Language.CZECH,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.HEBREW,
        Language.HINDI,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.MALAY,
        Language.DUTCH,
        Language.NORWEGIAN,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.SWAHILI,
        Language.TAMIL,
        Language.TELUGU,
        Language.THAI,
        Language.TURKISH,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
        Language.YORUBA,
        Language.ZULU,
    ]
    for formulation in [
        MCFFormulation(),
    ]
    for sensitivity_label in ["ALL"]
]

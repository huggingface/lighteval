"""
name:
Meta Mmlu

dataset:
meta-llama/Meta-Llama-3.1-8B-Instruct-evals

abstract:
Meta MMLU: A multilingual version of MMLU (using google translation)

languages:
french, german, hindi, italian, portuguese, spanish, thai

tags:
knowledge, multilingual, multiple-choice

paper:
https://arxiv.org/abs/2407.21783
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
        name=f"meta_mmlu_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["input_question"],
                "choices": [v for _, v in sorted(line["input_choice_list"].items(), key=lambda x: x[0])],
                "gold_idx": ascii_uppercase.index(line["input_correct_responses"][0]),
            },
            formulation=formulation,
        ),
        hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
        hf_subset=f"Llama-3.1-8B-Instruct-evals__multilingual_mmlu_{standardize_tag(language.value)}__details",
        hf_filter=partial(
            lambda language, subset, line: line["subtask_name"]
            == f"mmlu_{standardize_tag(language.value)}_chat.{subset}",
            language,
            subset,
        ),
        evaluation_splits=("latest",),
        hf_avail_splits=["latest"],
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
        Language.GERMAN,
        Language.SPANISH,
        Language.FRENCH,
        Language.HINDI,
        Language.ITALIAN,
        Language.PORTUGUESE,
        Language.THAI,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

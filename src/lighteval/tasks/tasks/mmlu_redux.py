"""
name:
Mmlu Redux

dataset:
edinburgh-dawg/mmlu-redux-2.0

abstract:
MMLU-Redux is a subset of 5,700 manually re-annotated questions across 57 MMLU subjects.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2406.04127
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


_MMLU_REDUX_2_SUBSETS = [
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
        name=f"mmlu_redux_2:{subset}",
        suite=["lighteval"],
        prompt_function=lambda line, task_name=None, s=subset: prompt.mmlu_redux_2(line, s, task_name),
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1,
        metrics=[
            Metrics.loglikelihood_acc,
            Metrics.pass_at_k_letters(sample_params={"k": 1}),
        ],
        stop_sequence=["\n"],
        version=0,
    )
    for subset in _MMLU_REDUX_2_SUBSETS
]

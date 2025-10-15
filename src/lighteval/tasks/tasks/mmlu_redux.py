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


_mmlu_redux_2_tasks = {
    subset: LightevalTaskConfig(
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
}

mmlu_redux_2_abstract_algebra = _mmlu_redux_2_tasks["abstract_algebra"]
mmlu_redux_2_anatomy = _mmlu_redux_2_tasks["anatomy"]
mmlu_redux_2_astronomy = _mmlu_redux_2_tasks["astronomy"]
mmlu_redux_2_business_ethics = _mmlu_redux_2_tasks["business_ethics"]
mmlu_redux_2_clinical_knowledge = _mmlu_redux_2_tasks["clinical_knowledge"]
mmlu_redux_2_college_biology = _mmlu_redux_2_tasks["college_biology"]
mmlu_redux_2_college_chemistry = _mmlu_redux_2_tasks["college_chemistry"]
mmlu_redux_2_college_computer_science = _mmlu_redux_2_tasks["college_computer_science"]
mmlu_redux_2_college_mathematics = _mmlu_redux_2_tasks["college_mathematics"]
mmlu_redux_2_college_medicine = _mmlu_redux_2_tasks["college_medicine"]
mmlu_redux_2_college_physics = _mmlu_redux_2_tasks["college_physics"]
mmlu_redux_2_computer_security = _mmlu_redux_2_tasks["computer_security"]
mmlu_redux_2_conceptual_physics = _mmlu_redux_2_tasks["conceptual_physics"]
mmlu_redux_2_econometrics = _mmlu_redux_2_tasks["econometrics"]
mmlu_redux_2_electrical_engineering = _mmlu_redux_2_tasks["electrical_engineering"]
mmlu_redux_2_elementary_mathematics = _mmlu_redux_2_tasks["elementary_mathematics"]
mmlu_redux_2_formal_logic = _mmlu_redux_2_tasks["formal_logic"]
mmlu_redux_2_global_facts = _mmlu_redux_2_tasks["global_facts"]
mmlu_redux_2_high_school_biology = _mmlu_redux_2_tasks["high_school_biology"]
mmlu_redux_2_high_school_chemistry = _mmlu_redux_2_tasks["high_school_chemistry"]
mmlu_redux_2_high_school_computer_science = _mmlu_redux_2_tasks["high_school_computer_science"]
mmlu_redux_2_high_school_european_history = _mmlu_redux_2_tasks["high_school_european_history"]
mmlu_redux_2_high_school_geography = _mmlu_redux_2_tasks["high_school_geography"]
mmlu_redux_2_high_school_government_and_politics = _mmlu_redux_2_tasks["high_school_government_and_politics"]
mmlu_redux_2_high_school_macroeconomics = _mmlu_redux_2_tasks["high_school_macroeconomics"]
mmlu_redux_2_high_school_mathematics = _mmlu_redux_2_tasks["high_school_mathematics"]
mmlu_redux_2_high_school_microeconomics = _mmlu_redux_2_tasks["high_school_microeconomics"]
mmlu_redux_2_high_school_physics = _mmlu_redux_2_tasks["high_school_physics"]
mmlu_redux_2_high_school_psychology = _mmlu_redux_2_tasks["high_school_psychology"]
mmlu_redux_2_high_school_statistics = _mmlu_redux_2_tasks["high_school_statistics"]
mmlu_redux_2_high_school_us_history = _mmlu_redux_2_tasks["high_school_us_history"]
mmlu_redux_2_high_school_world_history = _mmlu_redux_2_tasks["high_school_world_history"]
mmlu_redux_2_human_aging = _mmlu_redux_2_tasks["human_aging"]
mmlu_redux_2_human_sexuality = _mmlu_redux_2_tasks["human_sexuality"]
mmlu_redux_2_international_law = _mmlu_redux_2_tasks["international_law"]
mmlu_redux_2_jurisprudence = _mmlu_redux_2_tasks["jurisprudence"]
mmlu_redux_2_logical_fallacies = _mmlu_redux_2_tasks["logical_fallacies"]
mmlu_redux_2_machine_learning = _mmlu_redux_2_tasks["machine_learning"]
mmlu_redux_2_management = _mmlu_redux_2_tasks["management"]
mmlu_redux_2_marketing = _mmlu_redux_2_tasks["marketing"]
mmlu_redux_2_medical_genetics = _mmlu_redux_2_tasks["medical_genetics"]
mmlu_redux_2_miscellaneous = _mmlu_redux_2_tasks["miscellaneous"]
mmlu_redux_2_moral_disputes = _mmlu_redux_2_tasks["moral_disputes"]
mmlu_redux_2_moral_scenarios = _mmlu_redux_2_tasks["moral_scenarios"]
mmlu_redux_2_nutrition = _mmlu_redux_2_tasks["nutrition"]
mmlu_redux_2_philosophy = _mmlu_redux_2_tasks["philosophy"]
mmlu_redux_2_prehistory = _mmlu_redux_2_tasks["prehistory"]
mmlu_redux_2_professional_accounting = _mmlu_redux_2_tasks["professional_accounting"]
mmlu_redux_2_professional_law = _mmlu_redux_2_tasks["professional_law"]
mmlu_redux_2_professional_medicine = _mmlu_redux_2_tasks["professional_medicine"]
mmlu_redux_2_professional_psychology = _mmlu_redux_2_tasks["professional_psychology"]
mmlu_redux_2_public_relations = _mmlu_redux_2_tasks["public_relations"]
mmlu_redux_2_security_studies = _mmlu_redux_2_tasks["security_studies"]
mmlu_redux_2_sociology = _mmlu_redux_2_tasks["sociology"]
mmlu_redux_2_us_foreign_policy = _mmlu_redux_2_tasks["us_foreign_policy"]
mmlu_redux_2_virology = _mmlu_redux_2_tasks["virology"]
mmlu_redux_2_world_religions = _mmlu_redux_2_tasks["world_religions"]

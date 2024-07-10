from typing import Literal

from ..utils.prompts import get_arc_prompt, get_hellaswag_prompt, get_m_truthfulqa_prompt, get_mmlu_prompt, get_winogrande_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class HellaSwagTrTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"hellaswag-tr",
            prompt_function=get_hellaswag_prompt("tr"),
            suite=("custom",),
            hf_repo="malhajar/hellaswag_tr-v0.2",
            hf_subset="default",
            evaluation_splits=("validation",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

class WinogradeTrTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"xwinograd-tr",
            suite=("custom",),
            prompt_function=get_winogrande_prompt("tr"),
            hf_repo="malhajar/winogrande-tr-v0.2",
            hf_subset="default",
            filter=lambda x: x["sentence"].count("_") == 1 and len(x["sentence"].split("_")[0].strip()) > 0,
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm, Metrics.loglikelihood_acc_norm_pmi),
        )

class ARCEasyTrTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"arc-tr",
            prompt_function=get_arc_prompt("tr", nested_choices=True),
            suite=("custom",),
            hf_repo="malhajar/arc-tr-v0.2",
            hf_subset="default",
            evaluation_splits=("test",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        
MMLU_SUBSETS = Literal[
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
    "world_religions"
]


class MMLUTaskTr(LightevalTaskConfig):
    def __init__(self, subset: MMLU_SUBSETS):
        super().__init__(
            name=f"mmlu-tr:{subset}",
            prompt_function=get_mmlu_prompt("tr", is_number_choice=True),
            suite=("custom",),
            hf_repo="malhajar/mmlu_tr-v0.2",
            hf_subset=subset,
            evaluation_splits=("test",),
            few_shots_split="dev",
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm, Metrics.loglikelihood_acc_norm_pmi),
        )
        

class TruthfulQATrTask(LightevalTaskConfig):
    def __init__(self, type: Literal["mc1", "mc2"]):
        super().__init__(
            name=f"truthfulqa-tr:{type}",
            prompt_function=get_m_truthfulqa_prompt("tr", type),
            suite=("custom",),
            hf_repo="malhajar/truthful_qa-tr-v0.2",
            hf_subset="default",
            evaluation_splits=("validation",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_pmi,
            )
        )
        

#TODO: ADD gms8k, but I doubt it will be useful now
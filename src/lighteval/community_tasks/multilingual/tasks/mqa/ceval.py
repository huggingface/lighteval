from typing import Literal
from ..utils.prompts import get_ceval_prompt
from lighteval.metrics.metrics import Metrics

from lighteval.tasks.lighteval_task import LightevalTaskConfig


# Sometimes I dream that ts objects :( would come to python one day
CEVAL_TASK_TYPE = Literal[
    "computer_network",
    "operating_system",
    "computer_architecture",
    "college_programming",
    "college_physics",
    "college_chemistry",
    "advanced_mathematics",
    "probability_and_statistics",
    "discrete_mathematics",
    "electrical_engineer",
    "metrology_engineer",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "middle_school_mathematics",
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "veterinary_medicine",
    "college_economics",
    "business_administration",
    "marxism",
    "mao_zedong_thought",
    "education_science",
    "teacher_qualification",
    "high_school_politics",
    "high_school_geography",
    "middle_school_politics",
    "middle_school_geography",
    "modern_chinese_history",
    "ideological_and_moral_cultivation",
    "logic",
    "law",
    "chinese_language_and_literature",
    "art_studies",
    "professional_tour_guide",
    "legal_professional",
    "high_school_chinese",
    "high_school_history",
    "middle_school_history",
    "civil_servant",
    "sports_science",
    "plant_protection",
    "basic_medicine",
    "clinical_medicine",
    "urban_and_rural_planner",
    "accountant",
    "fire_engineer",
    "environmental_impact_assessment_engineer",
    "tax_accountant",
    "physician",
]


class CEvalTask(LightevalTaskConfig):
    def __init__(self, task: CEVAL_TASK_TYPE, show_options: bool = False):
        super().__init__(
            name=f"ceval{'_options' if show_options else ''}:{task}",
            prompt_function=get_ceval_prompt("zh", show_options),
            suite=("custom",),
            hf_repo="ceval/ceval-exam",
            hf_subset=task,
            evaluation_splits=("val",),
            few_shots_split="dev",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
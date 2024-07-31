from ast import List
from typing import Literal

from ..utils.prompts import get_arabic_mmlu_prompt, get_cmllu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


AR_MMLU_TASK_TYPE = Literal[
    "driving_test",
    "high_geography",
    "high_history",
    "islamic_studies",
    "univ_accounting",
    "primary_general_knowledge",
    "univ_political_science",
    "primary_math",
    "middle_general_knowledge",
    "high_biology",
    "primary_natural_science",
    "high_economics",
    "middle_natural_science",
    "middle_geography",
    "primary_social_science",
    "middle_computer_science",
    "middle_islamic_studies",
    "primary_computer_science",
    "high_physics",
    "middle_social_science",
    "middle_civics",
    "high_computer_science",
    "general_knowledge",
    "high_civics",
    "prof_law",
    "high_islamic_studies",
    "primary_arabic_language",
    "high_arabic_language",
    "arabic_language_grammar",
    "primary_history",
    "middle_history",
    "univ_economics",
    "arabic_language_general",
    "univ_computer_science",
    "primary_islamic_studies",
    "primary_geography",
    "high_philosophy",
    "middle_arabic_language",
    "middle_economics",
    "univ_management",
]

class ArabicMMLUTask(LightevalTaskConfig):
    def __init__(self, task: AR_MMLU_TASK_TYPE):
        true_task_name = " ".join(t.capitalize() for t in task.split("_"))
        super().__init__(
            name=f"arabic_mmlu_native:{task}",
            prompt_function=get_arabic_mmlu_prompt("ar"),
            suite=("custom",),
            hf_repo="MBZUAI/ArabicMMLU",
            hf_subset="default",
            filter = lambda x: x["Subject"] == true_task_name,
            evaluation_splits=("test",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
            ),
        )

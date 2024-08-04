from ast import List
from typing import Literal

from ..utils.prompts import get_arabic_mmlu_prompt, get_cmllu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


AR_MMLU_TASK_TYPE = Literal[
    "Driving Test",
    "High Geography",
    "High History",
    "Islamic Studies",
    "Univ Accounting",
    "Primary General Knowledge",
    "Univ Political Science",
    "Primary Math",
    "Middle General Knowledge",
    "High Biology",
    "Primary Natural Science",
    "High Economics",
    "Middle Natural Science",
    "Middle Geography",
    "Primary Social Science",
    "Middle Computer Science",
    "Middle Islamic Studies",
    "Primary Computer Science",
    "High Physics",
    "Middle Social Science",
    "Middle Civics",
    "High Computer Science",
    "General Knowledge",
    "High Civics",
    "Prof Law",
    "High Islamic Studies",
    "Primary Arabic Language",
    "High Arabic Language",
    "Arabic Language (Grammar)",
    "Primary History",
    "Middle History",
    "Univ Economics",
    "Arabic Language (General)",
    "Univ Computer Science",
    "Primary Islamic Studies",
    "Primary Geography",
    "High Philosophy",
    "Middle Arabic Language",
    "Middle Economics",
    "Univ Management",
]

class ArabicMMLUTask(LightevalTaskConfig):
    def __init__(self, task: AR_MMLU_TASK_TYPE, max_query_length: int, limit: int):
        super().__init__(
            name=f"arabic_mmlu_native:{task.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
            prompt_function=get_arabic_mmlu_prompt("ar"),
            suite=("custom",),
            hf_repo="yazeed7/ArabicMMLU",
            filter=lambda line: len(line["Question"]) + len(line.get("Context", "")) < max_query_length,
            hf_subset=task,
            limit=limit,
            evaluation_splits=("test",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
            ),
        )

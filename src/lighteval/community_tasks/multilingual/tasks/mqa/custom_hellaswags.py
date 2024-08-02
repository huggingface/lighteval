from functools import partial
import json
from typing import Literal, get_args

from ..utils.prompts import (
    get_hellaswag_prompt,
)
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class CustomHellaswagTeluguTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"custom_hellaswag-te",
            prompt_function=get_hellaswag_prompt("te", use_activity_label=False),
            suite=("custom",),
            hf_repo="LightFury9/hellaswag-telugu",
            hf_subset="default",
            evaluation_splits=("valid",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
        

class CustomHellaswagThaiTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"custom_hellaswag-th",
            prompt_function=get_hellaswag_prompt("th", use_activity_label=True),
            suite=("custom",),
            hf_repo="HuggingFaceFW-Dev/hellaswag_thai",
            hf_subset="default",
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
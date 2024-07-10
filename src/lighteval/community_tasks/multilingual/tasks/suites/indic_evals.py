from typing import Literal
from ..utils.prompts import get_arc_prompt, get_hellaswag_prompt_full_ctx, get_indic_boolq_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class ARCIndTask(LightevalTaskConfig):
    def __init__(self, subset: Literal["easy", "challenge"]):
        hf_subset = f"ARC-{subset.capitalize()}"
        super().__init__(
            name=f"arc-hi:{subset}",
            prompt_function=get_arc_prompt("hi", nested_choices=True),
            suite=("custom",),
            hf_repo="ai4bharat/ai2_arc-hi",
            hf_subset=hf_subset,
            evaluation_splits=("test",),
            few_shots_split="validation",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

class HellaSwagIndTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"hellaswag-hi",
            prompt_function=get_hellaswag_prompt_full_ctx("hi"),
            suite=("custom",),
            hf_repo="ai4bharat/hellaswag-hi",
            hf_subset="default",
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        
class BoolQIndTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"boolq-hi",
            prompt_function=get_indic_boolq_prompt("hi"),
            suite=("custom",),
            hf_repo="ai4bharat/boolq-hi",
            hf_subset="default",
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_arc_prompt, get_hellaswag_prompt_full_ctx, get_indic_boolq_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class ARCIndTask(LightevalTaskConfig):
    def __init__(self, subset: Literal["easy", "challenge"]):
        hf_subset = f"ARC-{subset.capitalize()}"
        super().__init__(
            name=f"hi-arc:{subset}",
            prompt_function=get_arc_prompt("hi", nested_choices=True),
            suite=("custom",),
            hf_repo="ai4bharat/ai2_arc-hi",
            hf_subset=hf_subset,
            evaluation_splits=("test",),
            few_shots_split="validation",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
            ),
        )

class HellaSwagIndTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"hi-hellaswag",
            prompt_function=get_hellaswag_prompt_full_ctx("hi"),
            suite=("custom",),
            hf_repo="ai4bharat/hellaswag-hi",
            hf_subset="default",
            filter=lambda x: all(len(choice.strip()) > 0 for choice in x["endings"]),
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
        
class BoolQIndTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"hi-boolq",
            prompt_function=get_indic_boolq_prompt("hi"),
            suite=("custom",),
            hf_repo="ai4bharat/boolq-hi",
            hf_subset="default",
            evaluation_splits=("validation",),
            few_shots_split="train",
            generation_size=5,
            stop_sequence=["\n"],
            metric=(
                get_qa_metric("hi", "exact"),
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
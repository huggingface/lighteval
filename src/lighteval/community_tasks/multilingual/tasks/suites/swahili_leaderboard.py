from typing import Literal

from ..utils.prompts import get_arc_prompt, get_mmlu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class MMLUSwTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"mmlu-sw",
            prompt_function=get_mmlu_prompt("sw", is_number_choice=True, zero_based=False),
            suite=("custom",),
            hf_repo="Mollel/MMUL_sw",
            hf_subset="default",
            filter=lambda x: x["language"] == "sw",
            evaluation_splits=("train",),
            few_shots_split=None,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )


class ARCSwTask(LightevalTaskConfig):
    def __init__(self, subset: Literal["easy", "challenge"]):
        repo = f"Mollel/ARC_{subset.capitalize()}_SWH"
        super().__init__(
            name=f"arc-sw:{subset}",
            prompt_function=get_arc_prompt("sw", nested_choices=True),
            suite=("custom",),
            hf_repo=repo,
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        


TASKS = [
    ARCSwTask(subset="easy"),
    ARCSwTask(subset="challenge"),
    MMLUSwTask(),
]
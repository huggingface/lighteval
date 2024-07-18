from typing import Literal

from ..utils.prompts import get_arc_prompt, get_mmlu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class MMLUSwTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"mmlu-sw",
            prompt_function=get_mmlu_prompt("sw", is_number_choice=True, zero_based=True),
            suite=("custom",),
            hf_repo="Mollel/MMUL_sw",
            hf_subset="default",
            hf_revision="2c8e2c6f9ad54450dd15481037d1bd0c5cbd0d1f",
            filter=lambda x: x["language"] == "sw",
            evaluation_splits=("train",),
            few_shots_split=None,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )


class ARCSwTask(LightevalTaskConfig):
    def __init__(self, subset: Literal["easy", "challenge"]):
        repo = f"Mollel/ARC_{subset.capitalize()}_SWH"
        revision = "5347439d3193c8a0dabaab3819914bf076dc94d4" if subset == "easy" else "dc1df9df632d14c251594d9129fb833d2ca4429c"
        super().__init__(
            name=f"arc-sw:{subset}",
            prompt_function=get_arc_prompt("sw", nested_choices=True),
            hf_revision=revision,
            suite=("custom",),
            hf_repo=repo,
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        


TASKS = [
    ARCSwTask(subset="easy"),
    ARCSwTask(subset="challenge"),
    MMLUSwTask(),
]
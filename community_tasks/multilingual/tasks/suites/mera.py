# THere are in total of 21 eval tasks, but most of them are instruction based.


from typing import Literal

from ..utils.prompts import get_hellaswag_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class CheGeKa(LightevalTaskConfig):
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
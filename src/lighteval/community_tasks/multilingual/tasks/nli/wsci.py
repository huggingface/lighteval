
from typing import Literal
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from ..utils.prompts import get_wsci_prompt


class WSCITask(LightevalTaskConfig):
    def __init__(self, lang: Literal["th"]):
        self.lang = lang
        super().__init__(
            name=f"wsci-{lang}",
            prompt_function=get_wsci_prompt(lang),
            suite=("custom",),
            hf_repo="pakphum/winograd_th",
            hf_subset=f"default",
            evaluation_splits=("test",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace, Metrics.loglikelihood_acc_norm_pmi],
        )
from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

from ..utils.prompts import get_copa_prompt


LANGS = Literal["as", "bn", "en", "gom", "gu", "hi", "kn", "mai", "ml", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]


class XCopaIndicTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        subset = f"translation-{lang}"
        super().__init__(
            name=f"xcopa-{lang}",
            suite=("custom",),
            prompt_function=get_copa_prompt(lang),
            hf_repo="ai4bharat/IndicCOPA",
            hf_subset=subset,
            evaluation_splits=("test",),
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace, Metrics.loglikelihood_acc_norm_pmi),
            trust_dataset=True,
        )

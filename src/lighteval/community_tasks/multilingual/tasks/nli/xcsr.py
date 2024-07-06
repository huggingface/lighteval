from typing import Literal

from ..utils.prompts import get_m_xcsr_prompt, xcodah_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw", "vi", "zh"]


# TODO: Wait until they fix the test set in the meantime we use validation for evals
class XCODAHTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"x-codah-{lang}",
            prompt_function=xcodah_prompt,
            suite=("custom",),
            hf_repo="INK-USC/xcsr",
            hf_subset=f"X-CODAH-{lang}",
            evaluation_splits=("validation",),
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        )


class XCSQATask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"x-csqa-{lang}",
            prompt_function=get_m_xcsr_prompt(lang),
            suite=("custom",),
            hf_repo="INK-USC/xcsr",
            hf_subset=f"X-CSQA-{lang}",
            evaluation_splits=("validation",),
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        )
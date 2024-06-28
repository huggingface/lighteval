from typing import Literal

from community_tasks.multilingual.tasks.utils.prompts import get_m_xcsr_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw", "vi", "zh"]


# TODO: Wait until they fix the test set
class XCODAHTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"x-codah:{lang}",
            prompt_function=get_m_xcsr_prompt(lang),
            suite=("custom",),
            hf_repo="INK-USC/xcsr",
            hf_subset=f"X-CODAH-{lang}",
            evaluation_splits=("test",),
            few_shots_split="validation",
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        )


class XCSQATask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"x-csqa:{lang}",
            prompt_function=get_m_xcsr_prompt(lang),
            suite=("custom",),
            hf_repo="INK-USC/xcsr",
            hf_subset=f"X-CSQA-{lang}",
            evaluation_splits=("test",),
            few_shots_split="validation",
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        )

from typing import Literal

from community_tasks.multilingual.tasks.utils.prompts import get_m_arc_prompt, m_hellaswag_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["en", "ar", "zh", "ru", "fr", "hi", "te"]


class M_HellaSwagTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"m_hellaswag_{lang}",
            prompt_function=m_hellaswag_prompt,
            suite=("custom",),
            hf_repo="alexandrainst/m_hellaswag",
            hf_subset=lang,
            evaluation_splits=("val",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_acc_norm_nospace_pmi,
            ),
        )


# TODO define the few-shot split
class M_MMLUTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"m_mmlu_{lang}",
            prompt_function=get_m_arc_prompt(lang),
            suite=("custom",),
            hf_repo="alexandrainst/m_mmlu",
            hf_subset=lang,
            evaluation_splits=("val",),
            generation_size=-1,
            stop_sequence=("\n",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )


class M_ARCTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"m_arc_{lang}",
            prompt_function=get_m_arc_prompt(lang),
            suite=("custom",),
            hf_repo="alexandrainst/m_arc",
            hf_subset=lang,
            evaluation_splits=("val",),
            generation_size=-1,
            stop_sequence=("\n",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

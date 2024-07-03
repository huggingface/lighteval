from typing import Literal

from ..utils.prompts import get_m_exams_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["bg", "hr", "hu", "it", "mk", "pl", "pt", "sq", "sr", "tr", "vi"]


# If too hard we can add help with para
class ExamsTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"exams-{lang}",
            prompt_function=get_m_exams_prompt(lang),
            suite=("custom",),
            hf_repo="mhardalov/exams",
            hf_subset=f"crosslingual_{lang}",
            evaluation_splits=("validation",),
            # Weird bug in dataset
            filter=lambda x: x["answerKey"] != "@",
            few_shots_split="train",
            generation_size=-1,
            stop_sequence=("\n",),
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace, Metrics.loglikelihood_acc_norm_pmi),
        )

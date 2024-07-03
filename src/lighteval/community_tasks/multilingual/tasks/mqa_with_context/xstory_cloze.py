from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks_prompt_formatting import storycloze


LANGS = Literal["zh", "ru", "en", "ar", "te", "sw", "hi"]


class XStoryClozeTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"xstory_cloze-{lang}",
            prompt_function=storycloze,
            hf_repo="juletxara/xstory_cloze",
            hf_subset=lang,
            hf_avail_splits=("training", "eval"),
            evaluation_splits=["eval"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=-1,
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm, Metrics.loglikelihood_acc_norm_pmi),
            stop_sequence=["\n"],
            version=0,
        )

from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]


class XCopaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"xcopa:{lang}",
            suite=("custom",),
            prompt_function=f"xcopa_{lang}",
            hf_repo="xcopa",
            hf_subset=lang,
            evaluation_splits=("test",),
            few_shots_split=None,
            few_shots_select=None,
            generation_size=-1,
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace),
            stop_sequence=("\n",),
            trust_dataset=True,
            version=0,
        )

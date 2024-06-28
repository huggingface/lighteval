from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks_prompt_formatting import winogrande


LANGS = Literal["zh", "ru", "fr", "en"]


class XWinogradeTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"xwinograd:{lang}",
            suite=("custom",),
            prompt_function=winogrande,
            hf_repo="Muennighoff/xwinograd",
            hf_subset=lang,
            evaluation_splits=("test",),
            few_shots_split=None,
            few_shots_select=None,
            generation_size=-1,
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm, Metrics.loglikelihood_acc_norm_pmi),
            stop_sequence=("\n",),
            output_regex=None,
            version=0,
        )

from typing import Literal

from ..utils.prompts import get_paws_x_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["de", "en", "es", "fr", "ja", "ko", "zh"]


class PawnsXTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"pawns-{lang}",
            suite=("custom",),
            prompt_function=get_paws_x_prompt(lang),
            hf_repo="google-research-datasets/paws-x",
            hf_subset=lang,
            evaluation_splits=("test",),
            few_shots_split="train",
            few_shots_select=None,
            generation_size=-1,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

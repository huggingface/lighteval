from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_mintaka_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# TODO Support all available
LANGS = Literal["ar", "de", "en", "es", "fr", "hi", "it", "ja", "pt"]


class MintakaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"mintaka-{lang}",
            prompt_function=get_mintaka_prompt(lang),
            suite=("custom",),
            hf_repo="AmazonScience/mintaka",
            hf_subset=lang,
            evaluation_splits=("test",),
            few_shots_split="train",
            generation_size=100,
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
            stop_sequence=("\n",),
        )

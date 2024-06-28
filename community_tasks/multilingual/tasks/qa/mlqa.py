from typing import Literal

from community_tasks.multilingual.tasks.utils.metrics import get_qa_metric
from community_tasks.multilingual.tasks.utils.prompts import get_mlqa_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# TODO Support all available
LANGS = Literal["zh", "en", "ar", "hi"]


class MlqaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.source_lang = lang
        super().__init__(
            name=f"mlqa:{lang}",
            prompt_function=get_mlqa_prompt(lang),
            suite=("custom",),
            hf_repo="facebook/mlqa",
            hf_subset=f"mlqa.{lang}.{lang}",
            evaluation_splits=("test",),
            few_shots_split="train",
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
        )

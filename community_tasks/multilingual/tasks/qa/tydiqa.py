# TODO Support all available
from typing import Literal

from community_tasks.multilingual.tasks.utils.metrics import get_qa_metric
from community_tasks.multilingual.tasks.utils.prompts import get_mlqa_prompt
from community_tasks.multilingual.tasks.utils.translation_literals import LANG_NAMES
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["zh", "en", "ar", "hi"]


class TydiqaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"tydiqa:{lang}",
            prompt_function=get_mlqa_prompt(lang),
            suite=("custom",),
            hf_repo="google-research-datasets/tydiqa",
            hf_subset="secondary_task",
            filter=lambda x: x["id"].split("-")[0] == LANG_NAMES[lang],
            evaluation_splits=("validation",),
            few_shots_split="train",
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
        )

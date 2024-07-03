from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_mlqa_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig


EVAL_TYPE = Literal["exact", "f1"]
LANGS = Literal["ar", "en", "hi", "ru", "th", "tr", "zh"]


class XquadTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"xquad-{lang}",
            prompt_function=get_mlqa_prompt(lang),
            suite=("custom",),
            hf_repo="google/xquad",
            hf_subset=f"xquad.{lang}",
            evaluation_splits=("validation",),
            few_shots_split="validation",
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
        )

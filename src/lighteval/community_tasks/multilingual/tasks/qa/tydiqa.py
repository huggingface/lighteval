# TODO Support all available
from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_mlqa_prompt
from ..utils.translation_literals import LANG_NAMES, LANG_NAMES_INVERTED
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["zh", "en", "ar", "hi", "te", "th", "sw", "ru"]


class TydiqaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS, max_query_length: int | None = None):
        super().__init__(
            name=f"tydiqa-{lang}",
            prompt_function=get_mlqa_prompt(lang),
            suite=("custom",),
            hf_repo="google-research-datasets/tydiqa",
            hf_subset="secondary_task",
            hf_revision="824c1b749da46e73930be9142d3b6815f2dded02",
            trust_dataset=True,
            filter=lambda x: x["id"].split("-")[0] == LANG_NAMES_INVERTED[lang] and (len(x["question"] + x["context"]) < max_query_length if max_query_length else True),
            evaluation_splits=("validation",),
            few_shots_split="train",
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
        )

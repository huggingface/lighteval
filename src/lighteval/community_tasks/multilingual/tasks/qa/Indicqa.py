

from typing import Literal
from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_mlqa_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

LANGS = Literal["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]


class IndicQATask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"indicqa.{lang}",
            prompt_function=get_mlqa_prompt(lang),
            suite=("custom",),
            hf_repo="ai4bharat/IndicQA",
            hf_subset=f"indicqa.{lang}",
            evaluation_splits=("test",),
            few_shots_split="test",
            filter=lambda x: all(len(a) != 0 for a in x["answers"]["text"]),
            generation_size=100,
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
            stop_sequence=("\n",),
        )
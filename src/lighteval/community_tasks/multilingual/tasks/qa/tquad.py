from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_tquad_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class Tquad2Task(LightevalTaskConfig):
    def __init__(self, max_query_length: int):
        super().__init__(
            name=f"tqduad2",
            prompt_function=get_tquad_prompt("tr"),
            suite=("custom",),
            hf_repo="erdometo/tquad2",
            hf_subset="default",
            evaluation_splits=("validation",),
            few_shots_split="train",
            filter=lambda x: len(x["question"] + x["context"]) < max_query_length,
            generation_size=60,
            stop_sequence=("\n",),
            metric=(get_qa_metric("tr", "exact"), get_qa_metric("tr", "f1")),
        )

from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_cmath_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig



class CMathTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"cmath",
            prompt_function=get_cmath_prompt("zh"),
            suite=("custom",),
            hf_repo="weitianwen/cmath",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="validation",
            generation_size=100,
            metric=(get_qa_metric("zh", "exact"), get_qa_metric("zh", "f1")),
            stop_sequence=("\n",),
        )

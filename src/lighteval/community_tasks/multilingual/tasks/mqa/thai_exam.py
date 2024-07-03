# https://github.com/stanford-crfm/helm/blob/b47a57a4e618b63d937bfac5a39aef9295beccab/src/helm/benchmark/scenarios/thai_exam_scenario.py#L10

from typing import Literal

from ..utils.prompts import get_m_exams_prompt, get_thai_exams_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


from typing import Literal

ThaiExamSubset = Literal["a_level", "ic", "onet", "tgat", "tpat1"]

# If too hard we can add help with para
class ThaiExamsTask(LightevalTaskConfig):
    def __init__(self, subset: ThaiExamSubset):
        super().__init__(
            name="thai-exams",
            prompt_function=get_thai_exams_prompt("th"),
            suite=("custom",),
            hf_repo="scb10x/thai_exam",
            hf_subset=subset,
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace, Metrics.loglikelihood_acc_norm_pmi),
        )

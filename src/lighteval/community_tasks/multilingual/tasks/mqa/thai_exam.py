# https://github.com/stanford-crfm/helm/blob/b47a57a4e618b63d937bfac5a39aef9295beccab/src/helm/benchmark/scenarios/thai_exam_scenario.py#L10

from typing import Literal

from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES

from ..utils.prompts import get_thai_exams_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


from typing import Literal

ThaiExamSubset = Literal["a_level", "ic", "onet", "tgat", "tpat1"]

# If too hard we can add help with para
class ThaiExamsTask(LightevalTaskConfig):
    def __init__(self, subset: ThaiExamSubset):

        def invalid_answers(line):
            pos_letters = [l.lower() for l in LETTER_INDICES[:5]]
            options = [line[letter] for letter in pos_letters if letter in line]
            non_empty_options = [str(opt) for opt in options]
            return all(opt.strip() != "" for opt in non_empty_options)
        

        super().__init__(
            name=f"thai-exams:{subset}",
            prompt_function=get_thai_exams_prompt("th"),
            suite=("custom",),
            hf_repo="scb10x/thai_exam",
            hf_subset=subset,
            filter=invalid_answers,
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi),
        )

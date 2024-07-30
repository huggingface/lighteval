
import re
from typing import Literal
from ..utils.prompts import get_agieval_prompt, answer_prefix_re, MULTICHOICE_JOIN_VARIANT
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


#TODO: CHECK FORMATTING OF THE PROMPT
CHINESE_AGIEVAL_TASK_TYPE = Literal[
    "gaokao-biology",
    "gaokao-chinese",
    "gaokao-chemistry",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-mathqa",
    "gaokao-physics",
    "logiqa-zh",
    "jec-qa-kd",
    "jec-qa-ca"
]

class ChineseAgievalTask(LightevalTaskConfig):
    def __init__(self, task: CHINESE_AGIEVAL_TASK_TYPE, show_options: bool = False, join_variant: MULTICHOICE_JOIN_VARIANT = "AND"):
        super().__init__(
            name=f"agieval{'_options' if show_options else ''}_{join_variant}:{task}",
            prompt_function=get_agieval_prompt("zh", show_options=show_options, join_variant=join_variant),
            suite=("custom",),
            hf_repo=f"hails/agieval-{task}",
            hf_subset="default",
            filter=lambda x: len(x["gold"]) > 0 and all(len(answer_prefix_re.sub("", choice)) > 0 for choice in x["choices"]),
            evaluation_splits=("test",),
            few_shots_split=None,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
            ),
        )

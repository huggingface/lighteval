
from typing import Literal
from ..utils.prompts import get_agieval_prompt
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
    def __init__(self, task: CHINESE_AGIEVAL_TASK_TYPE):
        super().__init__(
            name=f"agieval:{task}",
            prompt_function=get_agieval_prompt("zh"),
            suite=("custom",),
            hf_repo=f"hails/agieval-{task}",
            hf_subset="default",
            filter=lambda x: len(x["gold"]) > 0,
            evaluation_splits=("test",),
            few_shots_split=None,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

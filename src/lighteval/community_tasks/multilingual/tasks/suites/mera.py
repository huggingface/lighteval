# THere are in total of 21 eval tasks, but most of them are instruction based.


from typing import Literal

from ..utils.metrics import get_qa_metric

from ..utils.prompts import get_chekega_prompt, get_mathqa_prompt, get_openbookqa_prompt, get_parus_prompt, get_rcb_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class CheGeKaTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"chegeka",
            prompt_function=get_chekega_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="chegeka",
            evaluation_splits=("test",),
            few_shots_split="train",
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric("ru", "exact"), get_qa_metric("ru", "f1")),
        )

class MathLogicQATask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"math-logic-qa",
            prompt_function=get_mathqa_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="mathlogicqa",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )


class PARusTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"parus",
            prompt_function=get_parus_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="parus",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        
class RCBTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"rcb",
            prompt_function=get_rcb_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="rcb",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        

class RuMMLUTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"rummlu",
            prompt_function=get_mathqa_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="rummlu",
            evaluation_splits=("test",),
            few_shots_split="public_test",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

class RuOpenBookQATask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"rummlu",
            prompt_function=get_openbookqa_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="ruopenbookqa",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

class RuWorldTreeTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"ruworldtree",
            prompt_function=get_openbookqa_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="ruworldtree",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

# RWSD would be nice but it's hard to create a prompt
# POtentially use USE

_TASKS = [
    CheGeKaTask(),
    MathLogicQATask(),
    PARusTask(),
    RCBTask(),
    RuMMLUTask(),
    RuOpenBookQATask(),
    RuWorldTreeTask(),
]
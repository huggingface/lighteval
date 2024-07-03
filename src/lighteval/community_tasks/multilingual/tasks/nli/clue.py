


from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_c3_prompt, get_mlqa_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class CMRC2018Task(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"cmrc",
            prompt_function=get_mlqa_prompt("zh"),
            suite=("custom",),
            hf_repo="clue/clue",
            hf_subset="cmrc2018",
            evaluation_splits=("trial",),
            few_shots_split="train",
            generation_size=100,
            metric=(get_qa_metric("zh", "exact"), get_qa_metric("zh", "f1")),
            stop_sequence=("\n",),
        )

class C3Task(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"c3",
            prompt_function=get_c3_prompt("zh"),
            suite=("custom",),
            hf_repo="clue/clue",
            hf_subset="c3",
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        

# TODO: DEEPSEEK Also uses CHID + CCPM, but I am not sure how to prompt it
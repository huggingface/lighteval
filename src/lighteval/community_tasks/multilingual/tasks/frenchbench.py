from community_tasks.multilingual.common.prompts import (
    get_french_arc_prompt,
    get_french_boolqa_prompt,
    get_french_fquadv2_prompt,
)
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class FrenchARCTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french_bench:arc",
            prompt_function=get_french_arc_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french_bench_arc_challenge",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="train",
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        )


# Possible use "D'après l'information dans le contexte donné, quelle est la réponse à la question ?"
class FrenchBoolQATask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french_bench:boolqa",
            prompt_function=get_french_boolqa_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french_boolq",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="train",
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        )


class FrenchBenchFQuADv2Task(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french_bench:fquadv2",
            prompt_function=get_french_fquadv2_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/fquad2_test",
            hf_subset="default",
            evaluation_splits=("valid",),
            few_shots_split="train",
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.exact_match, Metrics.f1_score],
        )

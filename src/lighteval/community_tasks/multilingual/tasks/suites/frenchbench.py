from ..utils.metrics import get_qa_metric
from ..utils.prompts import (
    get_arc_prompt,
    get_french_boolqa_prompt,
    get_french_trivia_prompt,
    get_hellaswag_prompt,
    get_mlqa_prompt,
)
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class FrenchARCTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-arc",
            prompt_function=get_arc_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french_bench_arc_challenge",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

class FrenchHellaSwagTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-hellaswag",
            prompt_function=get_hellaswag_prompt("fr", use_activity_label=False),
            suite=("custom",),
            hf_repo="manu/french_bench_hellaswag",
            hf_subset="default",
            evaluation_splits=("validation",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

# Possible use "D'après l'information dans le contexte donné, quelle est la réponse à la question ?"
class BoolQAFrenchTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-boolqa",
            prompt_function=get_french_boolqa_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french_boolq",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="valid",
            generation_size=5,
            stop_sequence=["\n"],
            metric=(
                get_qa_metric("fr", "exact"),
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )


class FQuADv2Task(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="fquadv2",
            prompt_function=get_mlqa_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/fquad2_test",
            hf_subset="default",
            evaluation_splits=("test_hasAns",),
            few_shots_split="valid_hasAns",
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric("fr", "exact"), get_qa_metric("fr", "f1")),
        )
    

class TriviaFrenchTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-triviaqa",
            prompt_function=get_french_trivia_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french-trivia",
            hf_subset="default",
            evaluation_splits=("train",),
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric("fr", "exact"), get_qa_metric("fr", "f1")),
        )


# FrenchBench Fquad multi is a bit strange imo

_GENERATIVE_TASKS = [
    FQuADv2Task(),
    TriviaFrenchTask(),
    BoolQAFrenchTask(),
]

_MC_TASKS = [
    FrenchARCTask(),
    FrenchHellaSwagTask(),
    BoolQAFrenchTask(),
]
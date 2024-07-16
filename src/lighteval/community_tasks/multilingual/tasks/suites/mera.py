# THere are in total of 21 eval tasks, but most of them are instruction based.


from typing import Literal, get_args

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
            evaluation_splits=("train",),
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
            evaluation_splits=("train",),
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
            evaluation_splits=("train",),
            few_shots_split="validation",
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
            evaluation_splits=("train",),
            few_shots_split="validation",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )
        

RUMMLU_SUBSET = Literal[
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions"
]
class RuMMLUTask(LightevalTaskConfig):
    def __init__(self, subset: RUMMLU_SUBSET):
        super().__init__(
            name=f"rummlu:{subset}",
            prompt_function=get_mathqa_prompt("ru"),
            suite=("custom",),
            hf_repo="ai-forever/MERA",
            hf_subset="rummlu",
            filter=lambda x: x["meta"]["domain"] == subset,
            evaluation_splits=("public_test",),
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
            evaluation_splits=("train",),
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
            evaluation_splits=("train",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )

# RWSD would be nice but it's hard to create a prompt
# POtentially use USE

_RUMMLU_SUBSETS = [
    RuMMLUTask(subset) for subset in get_args(RUMMLU_SUBSET)
]

GENERATIVE_TASKS = [
    CheGeKaTask(),
]

MC_TASKS = [
    PARusTask(),
    MathLogicQATask(),
    RCBTask(),
    RuOpenBookQATask(),
    RuWorldTreeTask(),
    *_RUMMLU_SUBSETS,
]
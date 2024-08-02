from functools import partial
from typing import Literal, get_args

from ..utils.prompts import (
    get_arc_prompt,
    get_hellaswag_prompt,
    get_m_truthfulqa_prompt,
    get_mmlu_prompt,
)
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# trust_dataset is a bit scary and thus we lock the commit
LANGS = Literal["en", "ar", "zh", "ru", "fr", "hi", "te"]


class M_HellaSwagTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"hellaswag-{lang}",
            prompt_function=get_hellaswag_prompt(lang, use_activity_label=False),
            suite=("custom",),
            hf_repo="jon-tow/okapi_hellaswag",
            hf_subset=lang,
            hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",
            trust_dataset=True,
            evaluation_splits=("validation",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )


# TODO define the few-shot split
MMLU_SUBSET = Literal[
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
    "world_religions",
]

class M_MMLUTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS, subset: MMLU_SUBSET):
        super().__init__(
            name=f"mmlu-{lang}:{subset}",
            prompt_function=get_mmlu_prompt(lang),
            suite=("custom",),
            hf_repo="jon-tow/okapi_mmlu",
            hf_subset=lang,
            hf_revision="refs/pr/1",
            filter=lambda line: line["id"].split("/")[0] == subset,
            trust_dataset=True,
            evaluation_splits=("test",),
            few_shots_split="dev",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
        self.subset = subset



class M_ARCTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"arc-{lang}",
            prompt_function=get_arc_prompt(lang, nested_choices=True),
            suite=("custom",),
            hf_repo="jon-tow/okapi_arc_challenge",
            hf_subset=lang,
            hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
            trust_dataset=True,
            evaluation_splits=("test",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )


class M_TruthfulQATask(LightevalTaskConfig):
    def __init__(self, lang: LANGS, type: Literal["mc1", "mc2"]):
        super().__init__(
            name=f"truthfulqa-{lang}:{type}",
            prompt_function=get_m_truthfulqa_prompt(lang, type),
            suite=("custom",),
            hf_repo="jon-tow/okapi_truthfulqa",
            hf_subset=lang,
            hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
            trust_dataset=True,
            evaluation_splits=("validation",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
        


def get_mlmm_tasks(lang: LANGS):
    mmlu_tasks = [M_MMLUTask(lang, subset) for subset in get_args(MMLU_SUBSET)]
    return mmlu_tasks + [M_HellaSwagTask(lang), M_ARCTask(lang)]

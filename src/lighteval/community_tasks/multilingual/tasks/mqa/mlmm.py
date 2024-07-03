from typing import Literal

from ..utils.prompts import get_arc_prompt, get_hellaswag_prompt, get_m_truthfulqa_prompt, get_mmlu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# trust_dataset is a bit scary and thus we lock the commit
LANGS = Literal["en", "ar", "zh", "ru", "fr", "hi", "te"]


class M_HellaSwagTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"hellaswag-{lang}",
            prompt_function=get_hellaswag_prompt(lang),
            suite=("custom",),
            hf_repo="jon-tow/okapi_hellaswag",
            hf_subset=lang,
            hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",
            trust_dataset=True,
            evaluation_splits=("validation",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )


# TODO define the few-shot split
class M_MMLUTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"mmlu-{lang}",
            prompt_function=get_mmlu_prompt(lang),
            suite=("custom",),
            hf_repo="jon-tow/okapi_mmlu",
            hf_subset=lang,
            hf_revision="5d8c41172a1d463f718c793595308eb35f4fca02",
            trust_dataset=True,
            evaluation_splits=("test",),
            few_shots_split="dev",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi,
            ),
        )


class M_ARCTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"arc-{lang}",
            prompt_function=get_arc_prompt(lang),
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
                Metrics.loglikelihood_acc_norm_pmi,
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
                Metrics.loglikelihood_acc_norm_pmi,
            )
        )
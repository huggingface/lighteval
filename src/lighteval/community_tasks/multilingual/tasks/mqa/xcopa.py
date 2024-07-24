from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

from ..utils.prompts import get_copa_prompt


LANGS = Literal["ar", "et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]


class XCopaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        repo = "xcopa" if lang != "ar" else "OALL/AlGhafa-Arabic-LLM-Benchmark-Translated"
        subset = lang if lang != "ar" else "copa_ext_ar"
        #TODO: The ar also has fewshots
        super().__init__(
            name=f"xcopa-{lang}",
            suite=("custom",),
            prompt_function=get_copa_prompt(lang),
            hf_repo=repo,
            hf_subset=subset,
            evaluation_splits=("test",),
            few_shots_split=None,
            few_shots_select=None,
            generation_size=-1,
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi),
            stop_sequence=("\n",),
            trust_dataset=True,
            version=0,
        )

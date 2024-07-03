from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks_prompt_formatting import lambada


LANGS = Literal["en", "fr"]


class LambadaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"lambada-{lang}",
            prompt_function=lambada,
            suite=("custom",),
            hf_repo="EleutherAI/lambada_openai",
            hf_subset=lang,
            evaluation_splits=("test",),
            generation_size=10,
            stop_sequence=("\n",),
            output_regex=None,
            frozen=False,
            metric=[Metrics.target_perplexity],
        )

"""
name:
Chinese Squad

dataset:
lighteval/ChineseSquad

abstract:
ChineseSquad is a reading comprehension dataset for Chinese.

languages:
chinese

tags:
multilingual, qa

paper:
https://github.com/pluto-junzeng/ChineseSquad
"""

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"chinese_squad_{Language.CHINESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="lighteval/ChineseSquad",
        hf_subset="default",
        evaluation_splits=("validation",),
        few_shots_split="train",
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.CHINESE, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.CHINESE),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

"""
name:
Faquad

dataset:
eraldoluis/faquad

abstract:
FaQuAD: A Portuguese Reading Comprehension Dataset

languages:
portuguese

tags:
multilingual, qa

paper:
https://arxiv.org/abs/2007.15671
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
        name=f"faquad_{Language.PORTUGUESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.PORTUGUESE,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="eraldoluis/faquad",
        hf_subset="plain_text",
        hf_revision="205ba826a2282a4a5aa9bd3651e55ee4f2da1546",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("validation",),
        few_shots_split="train",
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.PORTUGUESE, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.PORTUGUESE),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

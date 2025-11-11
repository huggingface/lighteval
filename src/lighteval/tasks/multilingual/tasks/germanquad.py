"""
name:
Germanquad

dataset:
deepset/germanquad

abstract:
GermanQuAD: High-quality German QA dataset with 13,722 questions.

languages:
german

tags:
multilingual, qa

paper:
https://arxiv.org/abs/2104.12741
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
        name=f"germanquad_{Language.GERMAN.value}",
        prompt_function=get_qa_prompt_function(
            Language.GERMAN,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="deepset/germanquad",
        hf_subset="plain_text",
        hf_revision="fff05ceaf2ffbe5b65c7e0c57e678f7b7e1a0581",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.GERMAN, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.GERMAN),
        ),
    )
]

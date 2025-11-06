"""
name:
Drop Qa

dataset:
lighteval/drop_harness

abstract:
The DROP dataset is a new question-answering dataset designed to evaluate the
ability of language models to answer complex questions that require reasoning
over multiple sentences.

languages:
english

tags:
math, qa, reasoning

paper:
https://arxiv.org/abs/1810.00505
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


drop_qa = LightevalTaskConfig(
    name="drop",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "context": line["passage"],
            "question": line["question"],
            "choices": list(
                filter(
                    lambda x: x,
                    [line["answer"].get("number")]
                    + line["answer"]["spans"]
                    + [prompt.get_drop_date(line["answer"].get("date"))],
                )
            ),
        },
    ),
    hf_repo="lighteval/drop_harness",
    hf_subset="default",
    hf_filter=lambda line: list(
        filter(
            lambda x: x,
            [line["answer"].get("number")]
            + line["answer"]["spans"]
            + [prompt.get_drop_date(line["answer"].get("date"))],
        )
    ),
    evaluation_splits=("validation",),
    few_shots_split="train",
    generation_size=250,
    stop_sequence=["Question:", "question:", "\n"],
    metrics=[Metrics.exact_match],
    version=1,
)

TASKS_TABLE = [
    drop_qa,
]

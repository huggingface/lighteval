"""
name:
Jeopardy

dataset:
openaccess-ai-collective/jeopardy

abstract:
Jeopardy is a dataset of questions and answers from the Jeopardy game show.

languages:
english

tags:
knowledge, qa

paper:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


jeopardy = LightevalTaskConfig(
    name="jeopardy",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "choices": [line["answer"]],
        },
    ),
    hf_repo="openaccess-ai-collective/jeopardy",
    hf_subset="default",
    evaluation_splits=("train",),
    few_shots_split="train",
    generation_size=250,
    stop_sequence=["\n", "Question:", "question:"],
    metrics=[Metrics.exact_match],
    version=1,
)

TASKS_TABLE = [
    jeopardy,
]

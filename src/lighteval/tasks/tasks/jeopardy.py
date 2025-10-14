"""
abstract:
Jeopardy is a dataset of questions and answers from the Jeopardy game show.

languages:
en

paper:

tags:
knowledge, qa
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
    suite=("lighteval",),
    hf_repo="openaccess-ai-collective/jeopardy",
    hf_subset="default",
    evaluation_splits=("train",),
    few_shots_split="train",
    generation_size=250,
    stop_sequence=["\n", "Question:", "question:"],
    metrics=[Metrics.exact_match],
    version=1,
)

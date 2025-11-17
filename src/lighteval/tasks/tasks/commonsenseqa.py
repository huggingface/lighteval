"""
name:
Commonsenseqa

dataset:
tau/commonsense_qa

abstract:
CommonsenseQA is a new multiple-choice question answering dataset that requires
different types of commonsense knowledge to predict the correct answers . It
contains 12,102 questions with one correct answer and four distractor answers.
The dataset is provided in two major training/validation/testing set splits:
"Random split" which is the main evaluation split, and "Question token split",
see paper for details.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
https://arxiv.org/abs/1811.00937
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


commonsenseqa = LightevalTaskConfig(
    name="commonsenseqa",
    prompt_function=prompt.commonsense_qa,
    hf_repo="tau/commonsense_qa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    commonsenseqa,
]

"""
name:
Triviaqa

dataset:
mandarjoshi/trivia_qa

abstract:
TriviaqQA is a reading comprehension dataset containing over 650K
question-answer-evidence triples. TriviaqQA includes 95K question-answer pairs
authored by trivia enthusiasts and independently gathered evidence documents,
six per question on average, that provide high quality distant supervision for
answering the questions.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1705.03551
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


triviaqa = LightevalTaskConfig(
    name="triviaqa",
    prompt_function=prompt.triviaqa,
    hf_repo="mandarjoshi/trivia_qa",
    hf_subset="rc.nocontext",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", ".", ","],
    version=0,
)

TASKS_TABLE = [
    triviaqa,
]

"""
name:
Qasper

dataset:
allenai/qasper

abstract:
QASPER is a dataset for question answering on scientific research papers. It
consists of 5,049 questions over 1,585 Natural Language Processing papers. Each
question is written by an NLP practitioner who read only the title and abstract
of the corresponding paper, and the question seeks information present in the
full text. The questions are then answered by a separate set of NLP
practitioners who also provide supporting evidence to answers.

languages:
english

tags:
qa, scientific

paper:
https://arxiv.org/abs/2105.03011
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


qasper = LightevalTaskConfig(
    name="qasper",
    prompt_function=prompt.qasper,
    hf_repo="allenai/qasper",
    hf_subset="qasper",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.f1_score],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    qasper,
]

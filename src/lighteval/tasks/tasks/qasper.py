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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def qasper_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Title: {line['title']}\n\nPassage: {line['passage']}\n\n Question: {line['question']}\nAnswer: ",
        gold_index=0,
        choices=[line["gold"]],
    )


qasper = LightevalTaskConfig(
    name="qasper",
    prompt_function=qasper_prompt,
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

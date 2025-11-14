"""
name:
Asdiv

dataset:
EleutherAI/asdiv

abstract:
ASDiv is a dataset for arithmetic reasoning that contains 2,000+ questions
covering addition, subtraction, multiplication, and division.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2410.12853
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def asdiv_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['body']}\nQuestion:{line['question']}\nAnswer:",
        choices=line["answer"].split(" (")[0],
        gold_index=[0],
    )


asdiv = LightevalTaskConfig(
    name="asdiv",
    prompt_function=asdiv_prompt,
    hf_repo="EleutherAI/asdiv",
    hf_subset="asdiv",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [asdiv]

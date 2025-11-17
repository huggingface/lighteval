"""
name:
Lsat Qa

dataset:
lighteval/lsat_qa

abstract:
Questions from law school admission tests.

languages:
english

tags:
legal, qa

paper:
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def lsat_qa_prompt(line, task_name: str = None):
    query = f"The following are multiple choice questions (with answers).\nPassage: {line['passage']}\nQuestion: {line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["references"])])
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[: len(line["references"])],
        gold_index=line["gold_index"],
        instruction="The following are multiple choice questions (with answers).\n",
    )


lsat_qa = LightevalTaskConfig(
    name="lsat_qa",
    prompt_function=lsat_qa_prompt,
    hf_repo="lighteval/lsat_qa",
    hf_subset="all",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_assignment = LightevalTaskConfig(
    name="lsat_qa:assignment",
    prompt_function=lsat_qa_prompt,
    hf_repo="lighteval/lsat_qa",
    hf_subset="assignment",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_grouping = LightevalTaskConfig(
    name="lsat_qa:grouping",
    prompt_function=lsat_qa_prompt,
    hf_repo="lighteval/lsat_qa",
    hf_subset="grouping",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_miscellaneous = LightevalTaskConfig(
    name="lsat_qa:miscellaneous",
    prompt_function=lsat_qa_prompt,
    hf_repo="lighteval/lsat_qa",
    hf_subset="miscellaneous",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_ordering = LightevalTaskConfig(
    name="lsat_qa:ordering",
    prompt_function=lsat_qa_prompt,
    hf_repo="lighteval/lsat_qa",
    hf_subset="ordering",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    lsat_qa,
    lsat_qa_assignment,
    lsat_qa_grouping,
    lsat_qa_miscellaneous,
    lsat_qa_ordering,
]

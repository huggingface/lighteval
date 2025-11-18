"""
name:
Storycloze

dataset:
MoE-UNC/story_cloze

abstract:
A Corpus and Cloze Evaluation for Deeper Understanding of
Commonsense Stories

languages:
english

tags:
narrative, reasoning

paper:
https://arxiv.org/abs/1604.01696
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def storycloze_prompt(line, task_name: str = None):
    context = "\n".join(
        [line["input_sentence_1"], line["input_sentence_2"], line["input_sentence_3"], line["input_sentence_4"]]
    )
    choices = [line["sentence_quiz1"], line["sentence_quiz2"]]
    gold = int(line["answer_right_ending"]) - 1
    return Doc(task_name=task_name, query=context + "\n", choices=choices, gold_index=gold)


storycloze_2016 = LightevalTaskConfig(
    name="storycloze:2016",
    prompt_function=storycloze_prompt,
    hf_repo="MoE-UNC/story_cloze",
    hf_subset="2016",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


storycloze_2018 = LightevalTaskConfig(
    name="storycloze:2018",
    prompt_function=storycloze_prompt,
    hf_repo="MoE-UNC/story_cloze",
    hf_subset="2018",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    storycloze_2016,
    storycloze_2018,
]

"""
name:
Swag

dataset:
allenai/swag

abstract:
The dataset consists of 113k multiple choice questions about grounded situations
(73k training, 20k validation, 20k test). Each question is a video caption from
LSMDC or ActivityNet Captions, with four answer choices about what might happen
next in the scene. The correct answer is the (real) video caption for the next
event in the video; the three incorrect answers are adversarially generated and
human verified, so as to fool machines but not humans. SWAG aims to be a
benchmark for evaluating grounded commonsense NLI and for learning
representations.

languages:
english

tags:
narrative, reasoning

paper:
https://arxiv.org/abs/1808.05326
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def swag_prompt(line, task_name: str = None):
    choices = [line["ending0"], line["ending1"], line["ending2"], line["ending3"]]
    return Doc(
        task_name=task_name,
        query=line["startphrase"],
        choices=choices,
        gold_index=int(line["label"]),
    )


swag = LightevalTaskConfig(
    name="swag",
    prompt_function=swag_prompt,
    hf_repo="allenai/swag",
    hf_subset="regular",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    swag,
]

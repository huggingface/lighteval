"""
name:
Winogrande

dataset:
allenai/winogrande

abstract:
WinoGrande is a new collection of 44k problems, inspired by Winograd Schema
Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the
scale and robustness against the dataset-specific bias. Formulated as a
fill-in-a-blank task with binary options, the goal is to choose the right option
for a given sentence which requires commonsense reasoning.

languages:
english

tags:
commonsense, multiple-choice

paper:
https://arxiv.org/abs/1907.10641
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def winogrande_prompt(line, task_name: str = None):
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,
    )


winogrande = LightevalTaskConfig(
    name="winogrande",
    prompt_function=winogrande_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    winogrande,
]

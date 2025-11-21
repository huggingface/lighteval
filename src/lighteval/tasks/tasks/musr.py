"""
name:
Musr

dataset:
TAUR-Lab/MuSR

abstract:
MuSR is a benchmark for evaluating multistep reasoning in natural language
narratives. Built using a neurosymbolic synthetic-to-natural generation process,
it features complex, realistic tasks—such as long-form murder mysteries.

languages:
english

tags:
long-context, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2310.16049

starred:
true
"""

import ast
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def musr_prompt(line, task_name: str = None):
    choices = ast.literal_eval(line["choices"])

    query = line["narrative"] + "\n\n"
    query += line["question"] + "\n\n"
    for i, choice_ in enumerate(choices):
        query += f"{i + 1} - {choice_}\n"
    query += "Answer:"

    return Doc(task_name=task_name, query=query, choices=choices, gold_index=line["answer_index"])


def record_to_sample(record):
    query = record["narrative"] + "\n\n" + record["question"]
    choices = ast.literal_eval(record["choices"])
    target = ascii_uppercase[record["answer_index"]]
    return Sample(input=query, target=target, choices=choices)


musr_murder_mysteries = LightevalTaskConfig(
    name="musr:murder_mysteries",
    prompt_function=musr_prompt,
    hf_repo="TAUR-Lab/MuSR",
    hf_subset="default",
    hf_avail_splits=["murder_mysteries"],
    evaluation_splits=["murder_mysteries"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)


musr_object_placements = LightevalTaskConfig(
    name="musr:object_placements",
    prompt_function=musr_prompt,
    hf_repo="TAUR-Lab/MuSR",
    hf_subset="default",
    hf_avail_splits=["object_placements"],
    evaluation_splits=["object_placements"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)


musr_team_allocation = LightevalTaskConfig(
    name="musr:team_allocation",
    prompt_function=musr_prompt,
    hf_repo="TAUR-Lab/MuSR",
    hf_subset="default",
    hf_avail_splits=["team_allocation"],
    evaluation_splits=["team_allocation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    musr_murder_mysteries,
    musr_object_placements,
    musr_team_allocation,
]

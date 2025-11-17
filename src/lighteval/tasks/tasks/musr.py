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
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


musr_murder_mysteries = LightevalTaskConfig(
    name="musr:murder_mysteries",
    prompt_function=prompt.musr,
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
)


musr_object_placements = LightevalTaskConfig(
    name="musr:object_placements",
    prompt_function=prompt.musr,
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
)


musr_team_allocation = LightevalTaskConfig(
    name="musr:team_allocation",
    prompt_function=prompt.musr,
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
)

TASKS_TABLE = [
    musr_murder_mysteries,
    musr_object_placements,
    musr_team_allocation,
]

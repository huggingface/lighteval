# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

This module implements tasks for the french specific datasets
See : https://huggingface.co/fr-gouv-coordination-ia
"""

import random

import numpy as np
from aenum import extend_enum

import lighteval.tasks.extended.ifeval.instructions_registry as instructions_registry
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.extended.ifeval.main import ifeval_metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


# Ifeval-fr prompt function
def prompt_ifeval_fr(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


# qpqa-fr prompt function
def prompt_gpqa_fr(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Réponse incorrecte 1"], line["Réponse incorrecte 2"], line["Réponse incorrecte 3"]]
    choices.insert(gold_index, line["Réponse correcte"])

    instruction = "Choisissez la réponse correcte aux questions suivantes.\n\n"

    query = f"Question: {line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])
    query += "Réponse: "
    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


# BAC-fr prompt function
def prompt_bac_fr(line, task_name: str = None):
    prompt = f"Enoncé: {line['enonce']}\n{line['instruction']}\n"
    if line["choix"] is not None:  # Multichoice evaluation
        # prompt += "\n".join([f"{LETTER_INDICES[ix]}.{choix}" for ix, choix in enumerate(line["choix"])])
        return Doc(
            task_name=task_name,
            query=prompt,
            choices=as_list(line["choix"]),
            gold_index=line["choix"].index(line["choix correct"]),
            instruction="",
        )
    else:
        return Doc(task_name=task_name, query=prompt, choices=[line["reponse"]], gold_index=0, instruction="")


# IFEVal-fr task


ifeval_fr_task = LightevalTaskConfig(
    name="ifeval-fr",
    prompt_function=prompt_ifeval_fr,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="fr-gouv-coordination-ia/IFEval-fr",
    hf_subset="default",
    metric=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",  # select your metric in Metrics
)

# GPQA-fr task
gpqa_fr_task = LightevalTaskConfig(
    name="gpqa-fr",
    suite=["community"],
    prompt_function=prompt_gpqa_fr,
    hf_repo="fr-gouv-coordination-ia/gpqa-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# BAC-fr task
bac_fr_task = LightevalTaskConfig(
    name="bac-fr",
    suite=["community"],
    prompt_function=prompt_bac_fr,
    hf_repo="fr-gouv-coordination-ia/bac-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metric=[Metrics.quasi_exact_match_math, Metrics.exact_match],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [ifeval_fr_task, gpqa_fr_task, bac_fr_task]

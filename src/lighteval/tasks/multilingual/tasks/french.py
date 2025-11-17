"""
name:
French Evals

dataset:
fr-gouv-coordination-ia/IFEval-fr, fr-gouv-coordination-ia/gpqa-fr, fr-gouv-coordination-ia/bac-fr

abstract:
Collection of benchmarks for the french language.

languages:
french

tags:
knowledge, multiple-choice, qa

paper:
https://huggingface.co/fr-gouv-coordination-ia
"""

import random
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import math_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.ifeval.main import ifeval_metrics
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
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, choices)])
    query += "Réponse: "
    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        choices=ascii_uppercase[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


# BAC-fr prompt function
def prompt_bac_fr(line, task_name: str = None):
    prompt = f"Enoncé: {line['enonce']}\n{line['instruction']}\n"
    if line["choix"] is not None:  # Multichoice evaluation
        # prompt += "\n".join([f"{ascii_uppercase[ix]}.{choix}" for ix, choix in enumerate(line["choix"])])
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
    hf_repo="fr-gouv-coordination-ia/IFEval-fr",
    hf_subset="default",
    metrics=[ifeval_metrics],
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
    prompt_function=prompt_gpqa_fr,
    hf_repo="fr-gouv-coordination-ia/gpqa-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

# BAC-fr task
bac_fr_task = LightevalTaskConfig(
    name="bac-fr",
    prompt_function=prompt_bac_fr,
    hf_repo="fr-gouv-coordination-ia/bac-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.exact_match(sample_params={"normalize_gold": math_normalizer, "normalize_pred": math_normalizer}),
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [ifeval_fr_task, gpqa_fr_task, bac_fr_task]

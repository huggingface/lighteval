"""
name:
Covid Dialogue

dataset:
lighteval/covid_dialogue

abstract:
The COVID-19 Dialogue dataset is a collection of 500+ dialogues between
doctors and patients during the COVID-19 pandemic.

languages:
english

tags:
dialog, medical

paper:
https://arxiv.org/abs/2004.06561
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


PROMPT = "Generate a response given a patient's questions and concerns."


def covid_dialogue_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Generate a response given a patient's questions and concerns.\nPatient: {line['query']}\nDoctor: ",
        choices=[line["answer"]],
        gold_index=0,
        instruction="Generate a response given a patient's questions and concerns.\n",
    )


def record_to_sample(record):
    query = record["query"]
    target = record["answer"]
    return Sample(input=query, target=target)


covid_dialogue = LightevalTaskConfig(
    name="covid_dialogue",
    prompt_function=covid_dialogue_prompt,
    hf_repo="lighteval/covid_dialogue",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=128,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[system_message(PROMPT), generate(cache=True)],
    scorer=model_graded_fact(),
)

TASKS_TABLE = [
    covid_dialogue,
]

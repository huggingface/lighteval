"""
name:
Simpleqa

dataset:
lighteval/SimpleQA

abstract:
A factuality benchmark called SimpleQA that measures the ability for language
models to answer short, fact-seeking questions.

languages:
english

tags:
factuality, general-knowledge, qa

paper:
https://openai.com/index/introducing-simpleqa/

starred:
true
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def simpleqa_prompt(line, task_name: str = None):
    query = f"Question: {line['question']}\n"
    query += "".join(
        [f"\n{key}. {choice}" for key, choice in zip(["A", "B", "C", "D", "E", "F"], line["choices"]["text"])]
    )
    query += "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=line["choices"]["text"],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def record_to_sample(record):
    query = record["problem"]
    target = record["answer"]
    return Sample(input=query, target=target)


simpleqa = LightevalTaskConfig(
    name="simpleqa",
    prompt_function=simpleqa_prompt,
    hf_repo="lighteval/SimpleQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="few_shot",
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=model_graded_fact(),
)

TASKS_TABLE = [
    simpleqa,
]

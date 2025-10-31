"""
name:
MMLU Pro

dataset:
TIGER-Lab/MMLU-Pro

abstract:
MMLU-Pro dataset is a more robust and challenging massive multi-task
understanding dataset tailored to more rigorously benchmark large language
models' capabilities. This dataset contains 12K complex questions across various
disciplines.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2406.01574
"""
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

{choices}

Answer:""".strip()


def mmlu_pro_prompt_function(line, task_name: str = None):
    choices = "\n".join([f"{letter}: {choice}" for letter, choice in zip(ascii_uppercase, line["options"])])

    query = TEMPLATE.format(
        question=line["question"],
        choices=choices,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=ascii_uppercase[: len(choices)],
        gold_index=line["answer_index"],
        instruction=query,
    )


mmlu_pro = LightevalTaskConfig(
        name="mmlu_pro",
        prompt_function=mmlu_pro_prompt_function,
        suite=("lighteval",),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metrics=[Metrics.gpqa_instruct_metric],
    )

TASKS_TABLE = [mmlu_pro]

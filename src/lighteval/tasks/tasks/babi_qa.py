"""
name:
Babi Qa

dataset:
facebook/babi_qa

abstract:
The bAbI benchmark for measuring understanding and reasoning, evaluates reading
comprehension via question answering.

languages:
english

tags:
qa, reasoning

paper:
https://arxiv.org/abs/1502.05698
"""

import json

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# TODO: clean dataset and convert to inspect-ai


def babi_qa_prompt(line, task_name: str = None):
    def process_path(path: str) -> str:
        steps = path.split(",")
        directions = {"s": "south", "n": "north", "e": "east", "w": "west"}
        path = " ".join([directions[step] for step in steps])
        return path

    if isinstance(line["story"], dict):
        line = line["story"]
    else:
        line = json.loads(line["story"])

    results = []
    story = []
    for type, text, answer in zip(line["type"], line["text"], line["answer"]):
        if type == "supporting fact":
            story.append(text)
        elif type == "question":
            text = text.replace("_", process_path(answer))
            query = "\n".join(story) + f"\nQuestion: {text}\nAnswer: "
            results.append(Doc(task_name=task_name, query=query, choices=[answer], gold_index=0))
            story = []
    return results


babi_qa = LightevalTaskConfig(
    name="babi_qa",
    prompt_function=babi_qa_prompt,
    hf_repo="facebook/babi_qa",
    hf_subset="en-valid-qa1",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [babi_qa]

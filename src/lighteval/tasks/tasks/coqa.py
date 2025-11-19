"""
name:
Coqa

dataset:
stanfordnlp/coqa

abstract:
CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.

languages:
english

tags:
dialog, qa

paper:
https://arxiv.org/abs/1808.07042
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def coqa_prompt(line, task_name: str = None):
    results = []
    for q, a in zip(line["questions"], line["answers"]["input_text"]):
        results.append(Doc(task_name=task_name, query=f"{line['story']} \n\nQ: {q}\n\nA: ", choices=[a], gold_index=0))
    return results


coqa_first_question = LightevalTaskConfig(
    name="coqa",
    prompt_function=coqa_prompt,
    hf_repo="stanfordnlp/coqa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    stop_sequence=["\n", "Question:", "question:"],
    generation_size=100,
    version=1,
    metrics=[Metrics.exact_match],
)

TASKS_TABLE = [
    coqa_first_question,
]

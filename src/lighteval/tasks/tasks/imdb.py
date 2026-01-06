"""
name:
Imdb

dataset:
lighteval/IMDB_helm

abstract:
The IMDB benchmark for sentiment analysis in movie review, from:
Learning Word Vectors for Sentiment Analysis

languages:
english

tags:
classification

paper:
https://aclanthology.org/P11-1015/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def imdb_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['input']}\nSentiment: ",
        choices=["Positive", "Negative"],
        gold_index=["Positive", "Negative"].index(line["reference"]),
    )


def imdb_contrastset_prompt(line, task_name: str = None):
    if line["contrast_input"] is None or line["contrast_references"] is None:
        return imdb(line)

    return Doc(
        task_name=task_name,
        query=f"Passage: {line['contrast_inputs']}\nSentiment: ",
        choices=["Positive", "Negative"],
        gold_index=["Positive", "Negative"].index(line["contrast_references"]),
    )


imdb = LightevalTaskConfig(
    name="imdb",
    prompt_function=imdb_prompt,
    hf_repo="lighteval/IMDB_helm",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


imdb_contrastset = LightevalTaskConfig(
    name="imdb:contrastset",
    prompt_function=imdb_contrastset_prompt,
    hf_repo="lighteval/IMDB_helm",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    imdb,
    imdb_contrastset,
]

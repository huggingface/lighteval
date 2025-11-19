"""
name:
Summarization

dataset:
lighteval/summarization

abstract:
Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural
Networks for Extreme Summarization and: Abstractive Text Summarization using
Sequence-to-sequence RNNs and Beyond

languages:
english

tags:
summarization

paper:
https://aclanthology.org/D18-1206/
https://aclanthology.org/K16-1028/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def cnn_dm_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Article: {line['article']}\n\nTL;DR:",
        choices=[line["highlights"]],
        gold_index=0,
    )


def xsum_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Document: {line['document']}\n\nA one-sentence summary of the above document is:",
        choices=[line["summary"]],
        gold_index=0,
    )


summarization_cnn_dm = LightevalTaskConfig(
    name="summarization:cnn-dm",
    prompt_function=cnn_dm_prompt,
    hf_repo="lighteval/summarization",
    hf_subset="cnn-dm",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=128,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)


summarization_xsum = LightevalTaskConfig(
    name="summarization:xsum",
    prompt_function=xsum_prompt,
    hf_repo="lighteval/summarization",
    hf_subset="xsum",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=64,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)


summarization_xsum_sampled = LightevalTaskConfig(
    name="summarization:xsum-sampled",
    prompt_function=xsum_prompt,
    hf_repo="lighteval/summarization",
    hf_subset="xsum-sampled",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=64,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    summarization_cnn_dm,
    summarization_xsum,
    summarization_xsum_sampled,
]

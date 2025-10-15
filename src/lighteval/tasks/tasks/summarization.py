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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


summarization_cnn_dm = LightevalTaskConfig(
    name="summarization:cnn-dm",
    suite=["lighteval"],
    prompt_function=prompt.cnn_dm,
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
    suite=["lighteval"],
    prompt_function=prompt.xsum,
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
    suite=["lighteval"],
    prompt_function=prompt.xsum,
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

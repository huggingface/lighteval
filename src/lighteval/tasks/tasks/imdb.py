"""
abstract:
The IMDB benchmark for sentiment analysis in movie review, from:
Learning Word Vectors for Sentiment Analysis

languages:
en

tags:
sentiment-analysis

paper:
https://aclanthology.org/P11-1015/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


imdb = LightevalTaskConfig(
    name="imdb",
    suite=["lighteval"],
    prompt_function=prompt.imdb,
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
    suite=["lighteval"],
    prompt_function=prompt.imdb_contrastset,
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

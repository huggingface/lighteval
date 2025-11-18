"""
name:
Twitteraae

dataset:
lighteval/twitterAAE

abstract:
Demographic Dialectal Variation in Social Media: A Case Study of African-American English

languages:
english

tags:
language-modeling

paper:
https://aclanthology.org/D16-1120/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def twitter_aae_prompt(line, task_name: str = None):
    return Doc(task_name=task_name, query=line["tweet"], choices=None, gold_index=None)


twitterAAE_aa = LightevalTaskConfig(
    name="twitterAAE:aa",
    prompt_function=twitter_aae_prompt,
    hf_repo="lighteval/twitterAAE",
    hf_subset="aa",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)


twitterAAE_white = LightevalTaskConfig(
    name="twitterAAE:white",
    prompt_function=twitter_aae_prompt,
    hf_repo="lighteval/twitterAAE",
    hf_subset="white",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    twitterAAE_aa,
    twitterAAE_white,
]

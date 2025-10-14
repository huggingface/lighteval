"""
abstract:
Demographic Dialectal Variation in Social Media: A Case Study of African-American English

languages:
en

tags:
language-modeling

paper:
https://aclanthology.org/D16-1120/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


twitterAAE_aa = LightevalTaskConfig(
    name="twitterAAE:aa",
    suite=["lighteval"],
    prompt_function=prompt.twitter_aae,
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
    suite=["lighteval"],
    prompt_function=prompt.twitter_aae,
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

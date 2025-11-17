"""
name:
Wikitext

dataset:
EleutherAI/wikitext_document_level

abstract:
The WikiText language modeling dataset is a collection of over 100 million
tokens extracted from the set of verified Good and Featured articles on
Wikipedia. The dataset is available under the Creative Commons
Attribution-ShareAlike License.

languages:
english

tags:
language-modeling

paper:
https://arxiv.org/abs/1609.07843
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


wikitext_103_document_level = LightevalTaskConfig(
    name="wikitext:103:document_level",
    prompt_function=prompt.wikitext_helm,
    hf_repo="EleutherAI/wikitext_document_level",
    hf_subset="wikitext-103-raw-v1",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    wikitext_103_document_level,
]

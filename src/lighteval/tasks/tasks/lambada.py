"""
name:
Lambada

dataset:
cimec/lambada

abstract:
LAMBADA is a benchmark for testing language models’ ability to understand broad
narrative context. Each passage requires predicting its final word—easy for
humans given the full passage but impossible from just the last sentence.
Success demands long-range discourse comprehension.

languages:
english

tags:
language-modeling

paper:
https://arxiv.org/abs/1606.06031
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


lambada_standard = LightevalTaskConfig(
    name="lambada:standard",
    prompt_function=prompt.lambada,
    hf_repo="cimec/lambada",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.target_perplexity],
    stop_sequence=["\n"],
    version=0,
)


lambada_standard_cloze = LightevalTaskConfig(
    name="lambada:standard_cloze",
    prompt_function=prompt.lambada_cloze,
    hf_repo="cimec/lambada",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.target_perplexity],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    lambada_standard,
    lambada_standard_cloze,
]

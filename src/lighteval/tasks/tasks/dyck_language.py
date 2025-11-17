"""
name:
Dyck Language

dataset:
lighteval/DyckLanguage

abstract:
Scenario testing hierarchical reasoning through the Dyck formal languages.

languages:
english

tags:
reasoning

paper:
https://aclanthology.org/W19-3905/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


dyck_language_2 = LightevalTaskConfig(
    name="dyck_language:2",
    prompt_function=prompt.dyck_language,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="2",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


dyck_language_3 = LightevalTaskConfig(
    name="dyck_language:3",
    prompt_function=prompt.dyck_language,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="3",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


dyck_language_4 = LightevalTaskConfig(
    name="dyck_language:4",
    prompt_function=prompt.dyck_language,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="4",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    dyck_language_2,
    dyck_language_3,
    dyck_language_4,
]

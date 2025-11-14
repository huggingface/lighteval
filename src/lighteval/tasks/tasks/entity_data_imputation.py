"""
name:
Entity Data Imputation

dataset:
lighteval/Buy, lighteval/Restaurant

abstract:
Scenario that tests the ability to impute missing entities in a data table.

languages:
english

tags:
reasoning

paper:
https://ieeexplore.ieee.org/document/9458712
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def entity_data_imputation_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"What is the missing value?\n{line['text']}\nAnswer:",
        choices=[line["gold"]],
        gold_index=0,
        instruction="What is the missing value?\n",
    )


entity_data_imputation_Buy = LightevalTaskConfig(
    name="entity_data_imputation:Buy",
    prompt_function=entity_data_imputation_prompt,
    hf_repo="lighteval/Buy",
    hf_subset="default",
    hf_avail_splits=["train", "test", "valid"],
    evaluation_splits=["valid", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


entity_data_imputation_Restaurant = LightevalTaskConfig(
    name="entity_data_imputation:Restaurant",
    prompt_function=entity_data_imputation_prompt,
    hf_repo="lighteval/Restaurant",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
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
    entity_data_imputation_Buy,
    entity_data_imputation_Restaurant,
]

"""
name:
Entitymatching

dataset:
lighteval/EntityMatching

abstract:
Simple entity matching benchmark.

languages:
english

tags:
classification, reasoning

paper:
https://dl.acm.org/doi/10.14778/3007263.3007314
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def entity_matching_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Are Product A and Product B the same? Yes or No?\nProduct A is {line['productA']}. Product B is {line['productB']}. Are A and B the same?\nAnswer:",
        choices=["No", "Yes"],
        gold_index=int(line["same"]),
        instruction="Are Product A and Product B the same? Yes or No?\n",
    )


entity_matching_Abt_Buy = LightevalTaskConfig(
    name="entity_matching:Abt_Buy",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Abt_Buy",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Amazon_Google = LightevalTaskConfig(
    name="entity_matching:Amazon_Google",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Amazon_Google",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Beer = LightevalTaskConfig(
    name="entity_matching:Beer",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Beer",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Company = LightevalTaskConfig(
    name="entity_matching:Company",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Company",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_DBLP_ACM = LightevalTaskConfig(
    name="entity_matching:DBLP_ACM",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="DBLP_ACM",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_DBLP_GoogleScholar = LightevalTaskConfig(
    name="entity_matching:DBLP_GoogleScholar",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="DBLP_GoogleScholar",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Dirty_DBLP_ACM = LightevalTaskConfig(
    name="entity_matching:Dirty_DBLP_ACM",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Dirty_DBLP_ACM",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Dirty_DBLP_GoogleScholar = LightevalTaskConfig(
    name="entity_matching:Dirty_DBLP_GoogleScholar",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Dirty_DBLP_GoogleScholar",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Dirty_Walmart_Amazon = LightevalTaskConfig(
    name="entity_matching:Dirty_Walmart_Amazon",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Dirty_Walmart_Amazon",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Dirty_iTunes_Amazon = LightevalTaskConfig(
    name="entity_matching:Dirty_iTunes_Amazon",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Dirty_iTunes_Amazon",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Fodors_Zagats = LightevalTaskConfig(
    name="entity_matching=Fodors_Zagats",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Fodors_Zagats",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_Walmart_Amazon = LightevalTaskConfig(
    name="entity_matching:Walmart_Amazon",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="Walmart_Amazon",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

entity_matching_iTunes_Amazon = LightevalTaskConfig(
    name="entity_matching:iTunes_Amazon",
    prompt_function=entity_matching_prompt,
    hf_repo="lighteval/EntityMatching",
    hf_subset="iTunes_Amazon",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    entity_matching_Abt_Buy,
    entity_matching_Amazon_Google,
    entity_matching_Beer,
    entity_matching_Company,
    entity_matching_DBLP_ACM,
    entity_matching_DBLP_GoogleScholar,
    entity_matching_Dirty_DBLP_ACM,
    entity_matching_Dirty_DBLP_GoogleScholar,
    entity_matching_Dirty_Walmart_Amazon,
    entity_matching_Dirty_iTunes_Amazon,
    entity_matching_Fodors_Zagats,
    entity_matching_Walmart_Amazon,
    entity_matching_iTunes_Amazon,
]

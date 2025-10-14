"""
abstract:
Simple entity matching benchmark.

languages:
en

paper:
https://dl.acm.org/doi/10.14778/3007263.3007314
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


entity_matching_Abt_Buy = LightevalTaskConfig(
    name="entity_matching:Abt_Buy",
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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
    suite=["lighteval"],
    prompt_function=prompt.entity_matching,
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

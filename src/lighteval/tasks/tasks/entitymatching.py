# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


"""
abstract:
Simple entity matching benchmark.

languages:
en

paper:
https://dl.acm.org/doi/10.14778/3007263.3007314
"""

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

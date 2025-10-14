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
helm task
"""

raft_ade_corpus_v2 = LightevalTaskConfig(
    name="raft:ade_corpus_v2",
    suite=["lighteval"],
    prompt_function=prompt.raft_ade_corpus_v2,
    hf_repo="ought/raft",
    hf_subset="ade_corpus_v2",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_banking_77 = LightevalTaskConfig(
    name="raft:banking_77",
    suite=["lighteval"],
    prompt_function=prompt.raft_banking_77,
    hf_repo="ought/raft",
    hf_subset="banking_77",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_neurips_impact_statement_risks = LightevalTaskConfig(
    name="raft:neurips_impact_statement_risks",
    suite=["lighteval"],
    prompt_function=prompt.raft_neurips_impact_statement_risks,
    hf_repo="ought/raft",
    hf_subset="neurips_impact_statement_risks",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_one_stop_english = LightevalTaskConfig(
    name="raft:one_stop_english",
    suite=["lighteval"],
    prompt_function=prompt.raft_one_stop_english,
    hf_repo="ought/raft",
    hf_subset="one_stop_english",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_overruling = LightevalTaskConfig(
    name="raft:overruling",
    suite=["lighteval"],
    prompt_function=prompt.raft_overruling,
    hf_repo="ought/raft",
    hf_subset="overruling",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_semiconductor_org_types = LightevalTaskConfig(
    name="raft:semiconductor_org_types",
    suite=["lighteval"],
    prompt_function=prompt.raft_semiconductor_org_types,
    hf_repo="ought/raft",
    hf_subset="semiconductor_org_types",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_systematic_review_inclusion = LightevalTaskConfig(
    name="raft:systematic_review_inclusion",
    suite=["lighteval"],
    prompt_function=prompt.raft_systematic_review_inclusion,
    hf_repo="ought/raft",
    hf_subset="systematic_review_inclusion",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_tai_safety_research = LightevalTaskConfig(
    name="raft:tai_safety_research",
    suite=["lighteval"],
    prompt_function=prompt.raft_tai_safety_research,
    hf_repo="ought/raft",
    hf_subset="tai_safety_research",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_terms_of_service = LightevalTaskConfig(
    name="raft:terms_of_service",
    suite=["lighteval"],
    prompt_function=prompt.raft_terms_of_service,
    hf_repo="ought/raft",
    hf_subset="terms_of_service",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_tweet_eval_hate = LightevalTaskConfig(
    name="raft:tweet_eval_hate",
    suite=["lighteval"],
    prompt_function=prompt.raft_tweet_eval_hate,
    hf_repo="ought/raft",
    hf_subset="tweet_eval_hate",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_twitter_complaints = LightevalTaskConfig(
    name="raft:twitter_complaints",
    suite=["lighteval"],
    prompt_function=prompt.raft_twitter_complaints,
    hf_repo="ought/raft",
    hf_subset="twitter_complaints",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

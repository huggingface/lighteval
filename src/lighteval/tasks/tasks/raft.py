"""
name:
Raft

dataset:
ought/raft

abstract:
The Real-world annotated few-shot (RAFT) meta-benchmark of 11 real-world text
classification tasks.

languages:
english

tags:
classification, reasoning

paper:
https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ca46c1b9512a7a8315fa3c5a946e8265-Abstract-round2.html
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


raft_ade_corpus_v2 = LightevalTaskConfig(
    name="raft:ade_corpus_v2",
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

TASKS_TABLE = [
    raft_ade_corpus_v2,
    raft_banking_77,
    raft_neurips_impact_statement_risks,
    raft_one_stop_english,
    raft_overruling,
    raft_semiconductor_org_types,
    raft_systematic_review_inclusion,
    raft_tai_safety_research,
    raft_terms_of_service,
    raft_tweet_eval_hate,
    raft_twitter_complaints,
]

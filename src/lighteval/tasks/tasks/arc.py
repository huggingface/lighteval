"""
abstract:
7,787 genuine grade-school level, multiple-choice science questions, assembled
to encourage research in advanced question-answering. The dataset is partitioned
into a Challenge Set and an Easy Set, where the former contains only questions
answered incorrectly by both a retrieval-based algorithm and a word
co-occurrence algorithm

languages:
en

paper:
https://arxiv.org/abs/1803.05457
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


arc_challenge = LightevalTaskConfig(
    name="arc:challenge",
    suite=["lighteval"],
    prompt_function=prompt.arc,
    hf_repo="ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

arc_easy = LightevalTaskConfig(
    name="arc:easy",
    suite=["lighteval"],
    prompt_function=prompt.arc,
    hf_repo="ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

"""
name:
Synthetic Reasoning

dataset:
lighteval/synthetic_reasoning, lighteval/synthetic_reasoning_natural

abstract:
LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning

languages:
english

tags:
reasoning

paper:
https://arxiv.org/abs/2206.03855
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


synthetic_reasoning_induction = LightevalTaskConfig(
    name="synthetic_reasoning:induction",
    prompt_function=prompt.synthetic_reasoning,
    hf_repo="lighteval/synthetic_reasoning",
    hf_subset="induction",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=50,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


synthetic_reasoning_natural_easy = LightevalTaskConfig(
    name="synthetic_reasoning:natural_easy",
    prompt_function=prompt.synthetic_reasoning_natural,
    hf_repo="lighteval/synthetic_reasoning_natural",
    hf_subset="easy",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


synthetic_reasoning_natural_hard = LightevalTaskConfig(
    name="synthetic_reasoning:natural_hard",
    prompt_function=prompt.synthetic_reasoning_natural,
    hf_repo="lighteval/synthetic_reasoning_natural",
    hf_subset="hard",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


synthetic_reasoning_pattern_match = LightevalTaskConfig(
    name="synthetic_reasoning:pattern_match",
    prompt_function=prompt.synthetic_reasoning,
    hf_repo="lighteval/synthetic_reasoning",
    hf_subset="pattern_match",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=50,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


synthetic_reasoning_variable_substitution = LightevalTaskConfig(
    name="synthetic_reasoning:variable_substitution",
    prompt_function=prompt.synthetic_reasoning,
    hf_repo="lighteval/synthetic_reasoning",
    hf_subset="variable_substitution",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=50,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    synthetic_reasoning_induction,
    synthetic_reasoning_natural_easy,
    synthetic_reasoning_natural_hard,
    synthetic_reasoning_pattern_match,
    synthetic_reasoning_variable_substitution,
]

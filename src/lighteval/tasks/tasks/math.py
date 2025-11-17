"""
name:
Math

dataset:
DigitalLearningGmbH/MATH-lighteval

abstract:

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2305.20050
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import math_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig


math_algebra = LightevalTaskConfig(
    name="math:algebra",
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="algebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_counting_and_probability = LightevalTaskConfig(
    name="math:counting_and_probability",
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="counting_and_probability",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_geometry = LightevalTaskConfig(
    name="math:geometry",
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="geometry",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_intermediate_algebra = LightevalTaskConfig(
    name="math:intermediate_algebra",
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="intermediate_algebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_number_theory = LightevalTaskConfig(
    name="math:number_theory",
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="number_theory",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_prealgebra = LightevalTaskConfig(
    name="math:prealgebra",
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="prealgebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_precalculus = LightevalTaskConfig(
    name="math:precalculus",
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="precalculus",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

TASKS_TABLE = [
    math_algebra,
    math_counting_and_probability,
    math_geometry,
    math_intermediate_algebra,
    math_number_theory,
    math_prealgebra,
    math_precalculus,
]

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
The Mathematics Aptitude Test of Heuristics (MATH) dataset consists of problems
from mathematics competitions, including the AMC 10, AMC 12, AIME, and more.
Each problem in MATH has a full step-by-step solution, which can be used to
teach models to generate answer derivations and explanations.

languages:
en

tags:
math

paper:
https://arxiv.org/abs/2305.20050
"""

math_algebra = LightevalTaskConfig(
    name="math:algebra",
    suite=["lighteval"],
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="algebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

math_counting_and_probability = LightevalTaskConfig(
    name="math:counting_and_probability",
    suite=["lighteval"],
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="counting_and_probability",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

math_geometry = LightevalTaskConfig(
    name="math:geometry",
    suite=["lighteval"],
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="geometry",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

math_intermediate_algebra = LightevalTaskConfig(
    name="math:intermediate_algebra",
    suite=["lighteval"],
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="intermediate_algebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

math_number_theory = LightevalTaskConfig(
    name="math:number_theory",
    suite=["lighteval"],
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="number_theory",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

math_prealgebra = LightevalTaskConfig(
    name="math:prealgebra",
    suite=["lighteval"],
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="prealgebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

math_precalculus = LightevalTaskConfig(
    name="math:precalculus",
    suite=["lighteval"],
    prompt_function=prompt.math,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="precalculus",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

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
from: LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning

https://arxiv.org/abs/2206.03855
"""


synthetic_reasoning_induction = LightevalTaskConfig(
    name="synthetic_reasoning:induction",
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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

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
A small battery of 10 tests that involve asking language models a simple
arithmetic problem in natural language.
"""

arithmetic_1dc = LightevalTaskConfig(
    name="arithmetic:1dc",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_1dc",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_2da = LightevalTaskConfig(
    name="arithmetic:2da",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_2da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_2dm = LightevalTaskConfig(
    name="arithmetic:2dm",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_2dm",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_2ds = LightevalTaskConfig(
    name="arithmetic:2ds",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_2ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_3da = LightevalTaskConfig(
    name="arithmetic:3da",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_3da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_3ds = LightevalTaskConfig(
    name="arithmetic:3ds",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_3ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_4da = LightevalTaskConfig(
    name="arithmetic:4da",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_4da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_4ds = LightevalTaskConfig(
    name="arithmetic:4ds",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_4ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_5da = LightevalTaskConfig(
    name="arithmetic:5da",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_5da",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

arithmetic_5ds = LightevalTaskConfig(
    name="arithmetic:5ds",
    suite=["lighteval"],
    prompt_function=prompt.arithmetic,
    hf_repo="EleutherAI/arithmetic",
    hf_subset="arithmetic_5ds",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

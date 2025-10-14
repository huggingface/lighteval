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
Numeracy is a benchmark for evaluating the ability of language models to reason about mathematics.

languages:
en

tags:
math

paper:

"""

numeracy_linear_example = LightevalTaskConfig(
    name="numeracy:linear_example",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="linear_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_linear_standard = LightevalTaskConfig(
    name="numeracy:linear_standard",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="linear_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_parabola_example = LightevalTaskConfig(
    name="numeracy:parabola_example",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="parabola_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_parabola_standard = LightevalTaskConfig(
    name="numeracy:parabola_standard",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="parabola_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_paraboloid_example = LightevalTaskConfig(
    name="numeracy:paraboloid_example",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="paraboloid_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_paraboloid_standard = LightevalTaskConfig(
    name="numeracy:paraboloid_standard",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="paraboloid_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_plane_example = LightevalTaskConfig(
    name="numeracy:plane_example",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="plane_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_plane_standard = LightevalTaskConfig(
    name="numeracy:plane_standard",
    suite=["lighteval"],
    prompt_function=prompt.numeracy,
    hf_repo="lighteval/numeracy",
    hf_subset="plane_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

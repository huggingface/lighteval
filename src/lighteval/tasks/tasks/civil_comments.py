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
The CivilComments benchmark for toxicity detection.

languages:
en

tags:
toxicity, bias

paper:
https://arxiv.org/abs/1903.04561
"""

civil_comments = LightevalTaskConfig(
    name="civil_comments",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="all",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_LGBTQ = LightevalTaskConfig(
    name="civil_comments:LGBTQ",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="LGBTQ",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_black = LightevalTaskConfig(
    name="civil_comments:black",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="black",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_christian = LightevalTaskConfig(
    name="civil_comments:christian",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="christian",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_female = LightevalTaskConfig(
    name="civil_comments:female",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="female",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_male = LightevalTaskConfig(
    name="civil_comments:male",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="male",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_muslim = LightevalTaskConfig(
    name="civil_comments:muslim",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="muslim",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_other_religions = LightevalTaskConfig(
    name="civil_comments:other_religions",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="other_religions",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

civil_comments_white = LightevalTaskConfig(
    name="civil_comments:white",
    suite=["lighteval"],
    prompt_function=prompt.civil_comments,
    hf_repo="lighteval/civil_comments_helm",
    hf_subset="white",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

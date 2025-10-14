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
The Bias in Open-Ended Language Generation Dataset (BOLD) for measuring biases
and toxicity in open-ended language generation.

languages:
en

paper:
https://dl.acm.org/doi/10.1145/3442188.3445924
"""

bold = LightevalTaskConfig(
    name="bold",
    suite=["lighteval"],
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="all",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_gender = LightevalTaskConfig(
    name="bold:gender",
    suite=["lighteval"],
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="gender",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_political_ideology = LightevalTaskConfig(
    name="bold:political_ideology",
    suite=["lighteval"],
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="political_ideology",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_profession = LightevalTaskConfig(
    name="bold:profession",
    suite=["lighteval"],
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="profession",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_race = LightevalTaskConfig(
    name="bold:race",
    suite=["lighteval"],
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="race",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

bold_religious_ideology = LightevalTaskConfig(
    name="bold:religious_ideology",
    suite=["lighteval"],
    prompt_function=prompt.bold,
    hf_repo="lighteval/bold_helm",
    hf_subset="religious_ideology",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.prediction_perplexity],
    stop_sequence=["\n"],
    version=0,
)

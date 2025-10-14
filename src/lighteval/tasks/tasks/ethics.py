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
The Ethics benchmark for evaluating the ability of language models to reason about
ethical issues.

languages:
en

tags:
ethics, morality, commonsense, justice, utilitarianism, virtue

paper:
https://arxiv.org/abs/2008.02275
"""

ethics_commonsense = LightevalTaskConfig(
    name="ethics:commonsense",
    suite=["lighteval"],
    prompt_function=prompt.ethics_commonsense,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="commonsense",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_deontology = LightevalTaskConfig(
    name="ethics:deontology",
    suite=["lighteval"],
    prompt_function=prompt.ethics_deontology,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="deontology",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_justice = LightevalTaskConfig(
    name="ethics:justice",
    suite=["lighteval"],
    prompt_function=prompt.ethics_justice,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="justice",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_utilitarianism = LightevalTaskConfig(
    name="ethics:utilitarianism",
    suite=["lighteval"],
    prompt_function=prompt.ethics_utilitarianism,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="utilitarianism",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_virtue = LightevalTaskConfig(
    name="ethics:virtue",
    suite=["lighteval"],
    prompt_function=prompt.ethics_virtue,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="virtue",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

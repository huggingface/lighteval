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
Questions from law school admission tests.

languages:
en

tags:
legal, qa

paper:
"""

lsat_qa = LightevalTaskConfig(
    name="lsat_qa",
    suite=["lighteval"],
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="all",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_assignment = LightevalTaskConfig(
    name="lsat_qa:assignment",
    suite=["lighteval"],
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="assignment",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_grouping = LightevalTaskConfig(
    name="lsat_qa:grouping",
    suite=["lighteval"],
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="grouping",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_miscellaneous = LightevalTaskConfig(
    name="lsat_qa:miscellaneous",
    suite=["lighteval"],
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="miscellaneous",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lsat_qa_ordering = LightevalTaskConfig(
    name="lsat_qa:ordering",
    suite=["lighteval"],
    prompt_function=prompt.lsat_qa,
    hf_repo="lighteval/lsat_qa",
    hf_subset="ordering",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

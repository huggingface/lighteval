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
7,787 genuine grade-school level, multiple-choice science questions, assembled
to encourage research in advanced question-answering. The dataset is partitioned
into a Challenge Set and an Easy Set, where the former contains only questions
answered incorrectly by both a retrieval-based algorithm and a word
co-occurrence algorithm

from: Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge

https://arxiv.org/abs/1803.05457
"""

arc_challenge = LightevalTaskConfig(
    name="arc:challenge",
    suite=["lighteval"],
    prompt_function=prompt.arc,
    hf_repo="ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

arc_easy = LightevalTaskConfig(
    name="arc:easy",
    suite=["lighteval"],
    prompt_function=prompt.arc,
    hf_repo="ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

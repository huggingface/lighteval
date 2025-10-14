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
Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural
Networks for Extreme Summarization and: Abstractive Text Summarization using
Sequence-to-sequence RNNs and Beyond

languages:
en

tags:
summarization

paper:
https://aclanthology.org/D18-1206/
https://aclanthology.org/K16-1028/
"""

summarization_cnn_dm = LightevalTaskConfig(
    name="summarization:cnn-dm",
    suite=["lighteval"],
    prompt_function=prompt.cnn_dm,
    hf_repo="lighteval/summarization",
    hf_subset="cnn-dm",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=128,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)


summarization_xsum = LightevalTaskConfig(
    name="summarization:xsum",
    suite=["lighteval"],
    prompt_function=prompt.xsum,
    hf_repo="lighteval/summarization",
    hf_subset="xsum",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=64,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)


summarization_xsum_sampled = LightevalTaskConfig(
    name="summarization:xsum-sampled",
    suite=["lighteval"],
    prompt_function=prompt.xsum,
    hf_repo="lighteval/summarization",
    hf_subset="xsum-sampled",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=64,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)

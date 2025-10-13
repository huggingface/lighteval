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
Cant find this one in HELM, just a paper.

https://arxiv.org/abs/2210.13448
"""

legal_summarization_billsum = LightevalTaskConfig(
    name="legal_summarization:billsum",
    suite=["lighteval"],
    prompt_function=prompt.legal_summarization,
    hf_repo="lighteval/legal_summarization",
    hf_subset="BillSum",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
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


legal_summarization_eurlexsum = LightevalTaskConfig(
    name="legal_summarization:eurlexsum",
    suite=["lighteval"],
    prompt_function=prompt.legal_summarization,
    hf_repo="lighteval/legal_summarization",
    hf_subset="EurLexSum",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
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


legal_summarization_multilexsum = LightevalTaskConfig(
    name="legal_summarization:multilexsum",
    suite=["lighteval"],
    prompt_function=prompt.multilexsum,
    hf_repo="lighteval/legal_summarization",
    hf_subset="MultiLexSum",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
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

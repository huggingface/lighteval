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
OpenBookQA aims to promote research in advanced question-answering, probing a
deeper understanding of both the topic (with salient facts summarized as an open
book, also provided with the dataset) and the language it is expressed in. In
particular, it contains questions that require multi-step reasoning, use of
additional common and commonsense knowledge, and rich text comprehension.
OpenBookQA is a new kind of question-answering dataset modeled after open book
exams for assessing human understanding of a subject.


from: Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering
https://arxiv.org/abs/1809.02789
"""

openbookqa = LightevalTaskConfig(
    name="openbookqa",
    suite=["lighteval"],
    prompt_function=prompt.openbookqa_helm,
    hf_repo="openbookqa",
    hf_subset="main",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

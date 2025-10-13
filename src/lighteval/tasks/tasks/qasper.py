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
QASPER is a dataset for question answering on scientific research papers. It
consists of 5,049 questions over 1,585 Natural Language Processing papers. Each
question is written by an NLP practitioner who read only the title and abstract
of the corresponding paper, and the question seeks information present in the
full text. The questions are then answered by a separate set of NLP
practitioners who also provide supporting evidence to answers.

from: A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers
https://arxiv.org/abs/2105.03011
"""

qasper = LightevalTaskConfig(
    name="qasper",
    suite=["lighteval"],
    prompt_function=prompt.qasper,
    hf_repo="allenai/qasper",
    hf_subset="qasper",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.f1_score],
    stop_sequence=["\n"],
    version=0,
)

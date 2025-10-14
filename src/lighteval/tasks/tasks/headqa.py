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
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to
access a specialized position in the Spanish healthcare system, and are
challenging even for highly specialized humans. They are designed by the
Ministerio de Sanidad, Consumo y Bienestar Social, who also provides direct
access to the exams of the last 5 years.

languages:
en, es

tags:
health, reasoning

paper:
https://arxiv.org/abs/1906.04701
"""


headqa_en = LightevalTaskConfig(
    name="headqa:en",
    suite=["lighteval"],
    prompt_function=prompt.headqa,
    hf_repo="lighteval/headqa_harness",
    hf_subset="en",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)


headqa_es = LightevalTaskConfig(
    name="headqa:es",
    suite=["lighteval"],
    prompt_function=prompt.headqa,
    hf_repo="lighteval/headqa_harness",
    hf_subset="es",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)

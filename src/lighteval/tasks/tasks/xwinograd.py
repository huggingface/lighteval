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
Multilingual winograd schema challenge as used in Crosslingual Generalization through Multitask Finetuning.

languages:
en, fr, jp, pt, ru, zh

tags:
commonsense, commonsense-reasoning, multilingual

paper:
https://arxiv.org/abs/2211.01786
"""

xwinograd_en = LightevalTaskConfig(
    name="xwinograd:en",
    suite=["lighteval"],
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_fr = LightevalTaskConfig(
    name="xwinograd:fr",
    suite=["lighteval"],
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="fr",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_jp = LightevalTaskConfig(
    name="xwinograd:jp",
    suite=["lighteval"],
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="jp",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_pt = LightevalTaskConfig(
    name="xwinograd:pt",
    suite=["lighteval"],
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="pt",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_ru = LightevalTaskConfig(
    name="xwinograd:ru",
    suite=["lighteval"],
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xwinograd_zh = LightevalTaskConfig(
    name="xwinograd:zh",
    suite=["lighteval"],
    prompt_function=prompt.winogrande,
    hf_repo="Muennighoff/xwinograd",
    hf_subset="zh",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

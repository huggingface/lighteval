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
XStoryCloze consists of the professionally translated version of the English
StoryCloze dataset (Spring 2016 version) to 10 non-English languages. This
dataset is released by Meta AI.
"""

xstory_cloze_en_lighteval = LightevalTaskConfig(
    name="xstory_cloze:en",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="en",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_ru_lighteval = LightevalTaskConfig(
    name="xstory_cloze:ru",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="ru",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_zh_lighteval = LightevalTaskConfig(
    name="xstory_cloze:zh",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="zh",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_es_lighteval = LightevalTaskConfig(
    name="xstory_cloze:es",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="es",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_ar_lighteval = LightevalTaskConfig(
    name="xstory_cloze:ar",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="ar",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_hi_lighteval = LightevalTaskConfig(
    name="xstory_cloze:hi",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="hi",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_id_lighteval = LightevalTaskConfig(
    name="xstory_cloze:id",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="id",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_te_lighteval = LightevalTaskConfig(
    name="xstory_cloze:te",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="te",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_sw_lighteval = LightevalTaskConfig(
    name="xstory_cloze:sw",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="sw",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_eu_lighteval = LightevalTaskConfig(
    name="xstory_cloze:eu",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="eu",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xstory_cloze_my_lighteval = LightevalTaskConfig(
    name="xstory_cloze:my",
    suite=["lighteval"],
    prompt_function=prompt.storycloze,
    hf_repo="juletxara/xstory_cloze",
    hf_subset="my",
    hf_avail_splits=["training", "eval"],
    evaluation_splits=["eval"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

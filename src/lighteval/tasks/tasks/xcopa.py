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
XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning The Cross-lingual
Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability
of machine learning models to transfer commonsense reasoning across languages.

https://arxiv.org/abs/2005.00333
"""

xcopa_en = LightevalTaskConfig(
    name="xcopa:en",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_en,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="default",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_et = LightevalTaskConfig(
    name="xcopa:et",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_et,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="et",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_ht = LightevalTaskConfig(
    name="xcopa:ht",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_ht,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="ht",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_it = LightevalTaskConfig(
    name="xcopa:it",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_it,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="it",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_id = LightevalTaskConfig(
    name="xcopa:id",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_id,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="id",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_qu = LightevalTaskConfig(
    name="xcopa:qu",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_qu,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="qu",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_sw = LightevalTaskConfig(
    name="xcopa:sw",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_sw,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="sw",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_zh = LightevalTaskConfig(
    name="xcopa:zh",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_zh,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="zh",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_ta = LightevalTaskConfig(
    name="xcopa:ta",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_ta,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="ta",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_th = LightevalTaskConfig(
    name="xcopa:th",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_th,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="th",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_tr = LightevalTaskConfig(
    name="xcopa:tr",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_tr,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="tr",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

xcopa_vi = LightevalTaskConfig(
    name="xcopa:vi",
    suite=["lighteval"],
    prompt_function=prompt.xcopa_vi,
    hf_repo="cambridgeltl/xcopa",
    hf_subset="vi",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

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
The General Language Understanding Evaluation (GLUE) benchmark is a collection
of resources for training, evaluating, and analyzing natural language
understanding systems.

languages:
en

tags:


paper:
https://arxiv.org/abs/1804.07461
"""

glue_cola = LightevalTaskConfig(
    name="glue:cola",
    suite=["lighteval"],
    prompt_function=prompt.cola,
    hf_repo="glue",
    hf_subset="cola",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.mcc],
    stop_sequence=["\n"],
    version=0,
)

glue_mnli = LightevalTaskConfig(
    name="glue:mnli",
    suite=["lighteval"],
    prompt_function=prompt.mnli,
    hf_repo="glue",
    hf_subset="mnli_matched",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_mnli_mismatched = LightevalTaskConfig(
    name="glue:mnli_mismatched",
    suite=["lighteval"],
    prompt_function=prompt.mnli,
    hf_repo="glue",
    hf_subset="mnli_mismatched",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_mrpc = LightevalTaskConfig(
    name="glue:mrpc",
    suite=["lighteval"],
    prompt_function=prompt.mrpc,
    hf_repo="glue",
    hf_subset="mrpc",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],
    stop_sequence=["\n"],
    version=0,
)

glue_qnli = LightevalTaskConfig(
    name="glue:qnli",
    suite=["lighteval"],
    prompt_function=prompt.qnli,
    hf_repo="glue",
    hf_subset="qnli",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_qqp = LightevalTaskConfig(
    name="glue:qqp",
    suite=["lighteval"],
    prompt_function=prompt.qqp,
    hf_repo="glue",
    hf_subset="qqp",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],
    stop_sequence=["\n"],
    version=0,
)

glue_rte = LightevalTaskConfig(
    name="glue:rte",
    suite=["lighteval"],
    prompt_function=prompt.rte,
    hf_repo="glue",
    hf_subset="rte",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_sst2 = LightevalTaskConfig(
    name="glue:sst2",
    suite=["lighteval"],
    prompt_function=prompt.sst,
    hf_repo="glue",
    hf_subset="sst2",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_stsb = LightevalTaskConfig(
    name="glue:stsb",
    suite=["lighteval"],
    prompt_function=prompt.stsb,
    hf_repo="glue",
    hf_subset="stsb",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_wnli = LightevalTaskConfig(
    name="glue:wnli",
    suite=["lighteval"],
    prompt_function=prompt.wnli,
    hf_repo="glue",
    hf_subset="wnli",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_boolq = LightevalTaskConfig(
    name="super_glue:boolq",
    suite=["lighteval"],
    prompt_function=prompt.boolq_harness,
    hf_repo="super_glue",
    hf_subset="boolq",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_cb = LightevalTaskConfig(
    name="super_glue:cb",
    suite=["lighteval"],
    prompt_function=prompt.cb,
    hf_repo="super_glue",
    hf_subset="cb",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.multi_f1_numeric],
    stop_sequence=["\n"],
    version=0,
)

super_glue_copa = LightevalTaskConfig(
    name="super_glue:copa",
    suite=["lighteval"],
    prompt_function=prompt.copa,
    hf_repo="super_glue",
    hf_subset="copa",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_rte = LightevalTaskConfig(
    name="super_glue:rte",
    suite=["lighteval"],
    prompt_function=prompt.rte,
    hf_repo="super_glue",
    hf_subset="rte",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_multirc = LightevalTaskConfig(
    name="super_glue:multirc",
    suite=["lighteval"],
    prompt_function=prompt.multirc,
    hf_repo="super_glue",
    hf_subset="multirc",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_wic = LightevalTaskConfig(
    name="super_glue:wic",
    suite=["lighteval"],
    prompt_function=prompt.wic,
    hf_repo="super_glue",
    hf_subset="wic",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_wsc = LightevalTaskConfig(
    name="super_glue:wsc",
    suite=["lighteval"],
    prompt_function=prompt.wsc,
    hf_repo="super_glue",
    hf_subset="wsc",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

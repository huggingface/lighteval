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
BBQ: A hand-built bias benchmark for question answering

https://arxiv.org/abs/2110.08193
"""

bbq = LightevalTaskConfig(
    name="bbq",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="all",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Age = LightevalTaskConfig(
    name="bbq:Age",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Age",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Disability_status = LightevalTaskConfig(
    name="bbq:Disability_status",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Disability_status",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Gender_identity = LightevalTaskConfig(
    name="bbq:Gender_identity",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Gender_identity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Nationality = LightevalTaskConfig(
    name="bbq:Nationality",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Nationality",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Physical_appearance = LightevalTaskConfig(
    name="bbq:Physical_appearance",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Physical_appearance",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Race_ethnicity = LightevalTaskConfig(
    name="bbq:Race_ethnicity",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Race_ethnicity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Race_x_SES = LightevalTaskConfig(
    name="bbq:Race_x_SES",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Race_x_SES",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Race_x_gender = LightevalTaskConfig(
    name="bbq:Race_x_gender",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Race_x_gender",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Religion = LightevalTaskConfig(
    name="bbq:Religion",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Religion",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_SES = LightevalTaskConfig(
    name="bbq:SES",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="SES",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Sexual_orientation = LightevalTaskConfig(
    name="bbq:Sexual_orientation",
    suite=["lighteval"],
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Sexual_orientation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

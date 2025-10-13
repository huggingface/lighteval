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
MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the
true understanding capabilities of advanced AI models across multiple
modalities.

https://arxiv.org/abs/2409.02813
"""

mmmu_pro_standard_4_options = LightevalTaskConfig(
    name="mmmu_pro:standard-4",
    suite=["lighteval"],
    prompt_function=prompt.mmmu_pro,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="standard (4 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)


mmmu_pro_standard_10_options = LightevalTaskConfig(
    name="mmmu_pro:standard-10",
    suite=["lighteval"],
    prompt_function=prompt.mmmu_pro,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="standard (10 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)


mmmu_pro_vision = LightevalTaskConfig(
    name="mmmu_pro:vision",
    suite=["lighteval"],
    prompt_function=prompt.mmmu_pro_vision,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="vision",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)

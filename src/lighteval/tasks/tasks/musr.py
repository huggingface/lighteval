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
MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning

https://arxiv.org/abs/2310.16049
"""

musr_murder_mysteries = LightevalTaskConfig(
    name="musr:murder_mysteries",
    suite=["lighteval"],
    prompt_function=prompt.musr,
    hf_repo="TAUR-Lab/MuSR",
    hf_subset="default",
    hf_avail_splits=["murder_mysteries"],
    evaluation_splits=["murder_mysteries"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)


musr_object_placements = LightevalTaskConfig(
    name="musr:object_placements",
    suite=["lighteval"],
    prompt_function=prompt.musr,
    hf_repo="TAUR-Lab/MuSR",
    hf_subset="default",
    hf_avail_splits=["object_placements"],
    evaluation_splits=["object_placements"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)


musr_team_allocation = LightevalTaskConfig(
    name="musr:team_allocation",
    suite=["lighteval"],
    prompt_function=prompt.musr,
    hf_repo="TAUR-Lab/MuSR",
    hf_subset="default",
    hf_avail_splits=["team_allocation"],
    evaluation_splits=["team_allocation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

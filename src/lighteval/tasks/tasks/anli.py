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
from lighteval.metrics.metrics import Metrics, extractive_math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig, LightevalTaskConfig_inspect


anli_r1 = LightevalTaskConfig(
    name="anli:r1",
    suite=["lighteval"],
    prompt_function=prompt.anli,
    hf_repo="anli",
    hf_subset="plain_text",
    hf_avail_splits=["train_r1", "dev_r1", "test_r1"],
    evaluation_splits=["test_r1"],
    few_shots_split="train_r1",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)


anli_r2 = LightevalTaskConfig(
    name="anli:r2",
    suite=["lighteval"],
    prompt_function=prompt.anli,
    hf_repo="anli",
    hf_subset="plain_text",
    hf_avail_splits=["train_r2", "dev_r2", "test_r2"],
    evaluation_splits=["test_r2"],
    few_shots_split="train_r2",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)


anli_r3 = LightevalTaskConfig(
    name="anli:r3",
    suite=["lighteval"],
    prompt_function=prompt.anli,
    hf_repo="anli",
    hf_subset="plain_text",
    hf_avail_splits=["train_r3", "dev_r3", "test_r3"],
    evaluation_splits=["test_r3"],
    few_shots_split="train_r3",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

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


# aime24 = LightevalTaskConfig_inspect(
#     name="aime24",
#     prompt_function=prompt.aime_prompt_fn,
#     dataset_repo="HuggingFaceH4/aime_2024",
#     dataset_subset="default",
#     dataset_split="train",
#     scorers=[extractive_math_scorer()],
#     system_prompt="ASNWER USING THE FORMAT $ANSWER$",
#     epochs=16,
#     epochs_reducer="pass_at_4",
# )


# aime25 = LightevalTaskConfig_inspect(
#     name="aime25",
#     prompt_function=prompt.aime_prompt_fn,
#     dataset_repo="yentinglin/aime_2025",
#     dataset_subset="default",
#     dataset_split="train",
#     dataset_revision="main",
#     scorers=[extractive_math_scorer()],
#     system_prompt="ASNWER USING THE FORMAT $ANSWER$",
#     epochs=16,
#     epochs_reducer="pass_at_4",
# )


aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1})],
    version=2,
)

aime24_gpassk = LightevalTaskConfig(
    name="aime24_gpassk",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10000,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1})],
    version=2,
)

aime25_gpassk = LightevalTaskConfig(
    name="aime25_gpassk",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

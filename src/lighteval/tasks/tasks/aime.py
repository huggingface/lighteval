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


aime24 = LightevalTaskConfig_inspect(
    name="aime24",
    prompt_function=prompt.aime_prompt_fn,
    dataset_repo="HuggingFaceH4/aime_2024",
    dataset_subset="default",
    dataset_split="train",
    scorers=[extractive_math_scorer()],
    system_prompt="ASNWER USING THE FORMAT $ANSWER$",
    epochs=16,
    epochs_reducer="pass_at_4",
)


aime25 = LightevalTaskConfig_inspect(
    name="aime25",
    prompt_function=prompt.aime_prompt_fn,
    dataset_repo="yentinglin/aime_2025",
    dataset_subset="default",
    dataset_split="train",
    dataset_revision="main",
    scorers=[extractive_math_scorer()],
    system_prompt="ASNWER USING THE FORMAT $ANSWER$",
    epochs=16,
    epochs_reducer="pass_at_4",
)

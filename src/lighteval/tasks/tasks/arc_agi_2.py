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
ARC-AGI tasks are a series of three to five input and output tasks followed by a
final task with only the input listed. Each task tests the utilization of a
specific learned skill based on a minimal number of cognitive priors.
In their native form, tasks are a JSON lists of integers. These JSON can also be
represented visually as a grid of colors using an ARC-AGI task viewer. You can
view an example of a task here.
A successful submission is a pixel-perfect description (color and position) of
the final task's output.
100% of tasks in the ARC-AGI-2 dataset were solved by a minimim of 2 people in
less than or equal to 2 attempts (many were solved more). ARC-AGI-2 is more
difficult for AI.

languages:
en

paper:
https://arcprize.org/guide
"""

arc_agi_2 = LightevalTaskConfig(
    name="arc_agi_2",
    suite=["lighteval"],
    prompt_function=prompt.arc_agi_2,
    hf_repo="arc-agi-community/arc-agi-2",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    version=0,
)

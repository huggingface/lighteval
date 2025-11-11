"""
name:
ArcAgi 2

dataset:
arc-agi-community/arc-agi-2

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
english

tags:
multiple-choice

paper:
https://arcprize.org/guide
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


arc_agi_2 = LightevalTaskConfig(
    name="arc_agi_2",
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

TASKS_TABLE = [arc_agi_2]

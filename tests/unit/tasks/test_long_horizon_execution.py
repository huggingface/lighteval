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

import asyncio
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import Target

from lighteval.tasks.tasks.long_horizon_execution import scorer


TARGETS = [5, 12, 20]


def score_turns(turn_outputs):
    state = SimpleNamespace(
        metadata={"all_turn_outputs": turn_outputs, "targets_per_turn": TARGETS},
        output=SimpleNamespace(completion=turn_outputs[-1]),
    )
    return asyncio.run(scorer()(state, Target(str(TARGETS[-1])))).value


@pytest.mark.parametrize(
    "turn_outputs, turn_accuracy, horizon, all_correct",
    [
        (["<answer>5</answer>", "<answer>12</answer>", "<answer>20</answer>"], 1.0, 3, True),
        (["The sum is 5", "<answer>12</answer>", "<answer>20</answer>"], 2 / 3, 0, False),
        (["<answer>5</answer>", "The sum is 12", "<answer>20</answer>"], 2 / 3, 1, False),
        (["<answer>5</answer>", "<answer>twelve</answer>", "<answer>20</answer>"], 2 / 3, 1, False),
        (["The sum is 5", "The sum is 12", "The sum is 20"], 0.0, 0, False),
    ],
    ids=["all_tagged", "first_turn_untagged", "middle_turn_untagged", "middle_turn_not_a_number", "no_tags"],
)
def test_turn_without_answer_tag_keeps_later_turns_aligned(turn_outputs, turn_accuracy, horizon, all_correct):
    value = score_turns(turn_outputs)

    assert value["turn_accuracy"] == pytest.approx(turn_accuracy)
    assert value["horizon"] == horizon
    assert value["all_correct"] is all_correct

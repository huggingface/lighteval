# MIT License

# Copyright (c) 2026 The HuggingFace Team

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

import pytest

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.aimo import aimo_prompt, task


# Problem 229ee8 from lighteval/aimo_progress_prize_1
LINE = {
    "problem": "Let $k, l > 0$ be parameters. The parabola $y = kx^2 - 2kx + l$ intersects the line $y = 4$ at two "
    "points $A$ and $B$. These points are distance 6 apart. What is the sum of the squares of the distances from "
    "$A$ and $B$ to the origin?",
    "answer": 52,
}


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("The sum of the squares of the distances is $\\boxed{52}$.", 1),
        ("The sum of the squares of the distances is $\\boxed{40}$.", 0),
        ("The sum of the squares of the distances is 40.", 0),
    ],
)
def test_aimo_exact_match(prediction, expected):
    doc = aimo_prompt(LINE, task_name="aimo_progress_prize_1")
    model_response = ModelResponse(text=[prediction])

    for metric in task.metrics:
        result = metric.compute_sample(doc=doc, model_response=model_response)
        assert result[metric.metric_name] == expected

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

import pytest

from lighteval.metrics.metrics_sample import LoglikelihoodAcc, Recall
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


@pytest.mark.parametrize(
    "logprobs, gold_index, expected",
    [
        ([-1.0, -1.0, -1.0], 0, [1, 1, 1]),
        ([-1.0, -1.0, -1.0], 1, [0, 1, 1]),
        ([-1.0, -1.0, -1.0], 2, [0, 0, 1]),
        ([-1.0, -2.0, -2.0, -3.0], 1, [0, 1, 1, 1]),
        ([-1.0, -2.0, -2.0, -3.0], 2, [0, 0, 1, 1]),
        ([-2.0, -1.0, -1.0, -3.0], 1, [1, 1, 1, 1]),
        ([-3.0, -1.0, -2.0], 0, [0, 0, 1]),
        ([-3.0, -1.0, -2.0], 1, [1, 1, 1]),
        ([-3.0, -1.0, -2.0], 2, [0, 1, 1]),
        ([-1.0, -1.0, -1.0], [0, 2], [1, 1, 1]),
        ([-1.0, -2.0, -2.0, -3.0], [2, 3], [0, 0, 1, 1]),
        ([-1.0], 0, [1]),
        ([float("-inf"), float("-inf"), float("-inf")], 0, [1, 1, 1]),
    ],
)
def test_recall_ranking(logprobs, gold_index, expected):
    doc = Doc(query="q", choices=[str(i) for i in range(len(logprobs))], gold_index=gold_index)
    response = ModelResponse(logprobs=logprobs)

    scores = [Recall(k).compute(doc, response) for k in range(1, len(logprobs) + 1)]

    assert scores == expected
    assert scores == sorted(scores)
    assert scores[0] == LoglikelihoodAcc().compute(doc, response)
    assert Recall(len(logprobs) + 1).compute(doc, response) == 1
    assert Recall(0).compute(doc, response) == 0

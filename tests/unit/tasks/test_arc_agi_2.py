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

from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.arc_agi_2 import arc_agi_2_prompt


RECORD = {
    "fewshots": [{"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]}],
    "question": [{"input": [[0, 0], [1, 0]], "output": [[0, 0], [0, 1]]}],
}


def exact_match(answer: str) -> float:
    doc = arc_agi_2_prompt(RECORD, "arc_agi_2")
    return Metrics.exact_match.value.sample_level_fn.compute(doc=doc, model_response=ModelResponse(text=[answer]))


def test_prompt_shows_example_outputs_one_row_per_line():
    doc = arc_agi_2_prompt(RECORD, "arc_agi_2")

    assert "OUTPUT: \n\n[0, 1]\n[0, 0]\n" in doc.query


@pytest.mark.parametrize(
    "answer",
    [
        "[0, 0]\n[0, 1]",  # the format of the example outputs in the prompt
        "[0, 0]\n[0, 1]\n",
        "[[0, 0], [0, 1]]",  # a one-line list still counts
    ],
)
def test_correct_grid_scores_in_the_prompt_format_and_as_a_list(answer):
    assert exact_match(answer) == 1


@pytest.mark.parametrize("answer", ["[0, 1]\n[0, 0]", "[[0, 1], [0, 0]]", "[0, 0]", ""])
def test_wrong_grid_scores_zero(answer):
    assert exact_match(answer) == 0

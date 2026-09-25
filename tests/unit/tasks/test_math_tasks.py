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
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.math import TASKS_TABLE


GOLD = " We have $x + 3 = 5$, so $x = \\boxed{2}$."


@pytest.mark.parametrize("task", TASKS_TABLE, ids=lambda t: t.name)
@pytest.mark.parametrize("answer, expected", [("\\boxed{2}", 1), ("\\boxed{3}", 0)])
def test_math_maj_at_n_applies_math_normalizer(task, answer, expected):
    metric = task.metrics[0].sample_level_fn
    doc = Doc(query="Question: Solve $x + 3 = 5$.\nAnswer:", choices=[GOLD], gold_index=0)
    response = ModelResponse(text=[f"Subtracting 3 gives $x = {answer}$."] * metric.n)

    assert metric.compute(doc=doc, model_response=response) == expected

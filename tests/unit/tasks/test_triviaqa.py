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
from lighteval.tasks.tasks.triviaqa import triviaqa, triviaqa_prompt


# Questions bb_5713 and wh_4325 from the TriviaQA rc.nocontext validation split
JOHN_NASH = {
    "question": "Name the pioneering Game Theory US mathematician featured in the film A Beautiful Mind?",
    "answer": {
        "aliases": [
            "John Nash (disambiguation)",
            "Johnny Nash (disambiguation)",
            "John nash",
            "John Nash",
            "Nash, John",
        ]
    },
}
VELAZQUEZ = {
    "question": "Give a year in the life of the artist Diego Velazquez.",
    "answer": {"aliases": ["1599-1660"]},
}


@pytest.mark.parametrize(
    "line, prediction, expected",
    [
        (JOHN_NASH, " John Nash", 1),
        (JOHN_NASH, " Johnny Nash", 0),
        (VELAZQUEZ, " 1599-1660", 1),
    ],
)
def test_triviaqa_exact_match_normalizes_prediction(line, prediction, expected):
    doc = triviaqa_prompt(line, task_name="triviaqa")
    model_response = ModelResponse(text=[prediction])

    for metric in triviaqa.metrics:
        result = metric.compute_sample(doc=doc, model_response=model_response)
        assert result[metric.metric_name] == expected

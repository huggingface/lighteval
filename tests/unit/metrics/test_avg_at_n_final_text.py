# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""avg@n must score the same text as every other sampling metric.

Two regressions are covered here, both of which return a plausible number rather
than raising:

1. `ModelResponse.__getitem__` rebuilt a single-generation response without
   `text_post_processed`, so `final_text` on the slice fell back to the raw
   generation. `AvgAtN` was the only metric reading a sliced response, so with
   `remove_reasoning_tags` enabled (the pipeline default) avg@n was graded
   against text that still contained the model's reasoning block while maj@n and
   pass@k on the identical generations were graded against the stripped answer.

2. `AvgAtN.compute` never called `self.preprocess`, so `strip_strings` and
   `normalize` were silently dropped even though they are declared on the
   registered metric.
"""

import pytest

from lighteval.metrics.metrics_sample import AvgAtN, MajAtN, PassAtK
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import remove_reasoning_tags


REASONING_TAGS = [("<think>", "</think>")]


def _doc() -> Doc:
    return Doc(query="What is the capital of France?", choices=["Paris"], gold_index=0, task_name="t")


def _response_with_reasoning(answers: list[str]) -> ModelResponse:
    """A response shaped the way Pipeline._post_process_outputs leaves it."""
    raw = [f"<think>Could be Lyon. No, it is Paris.</think>{answer}" for answer in answers]
    response = ModelResponse(text=raw)
    response.text_post_processed = [remove_reasoning_tags(text=t, tag_pairs=REASONING_TAGS) for t in raw]
    return response


def test_getitem_keeps_post_processed_text():
    response = _response_with_reasoning(["Paris"])
    assert response.final_text[0] == "Paris"
    assert response[0].final_text[0] == "Paris"


def test_avg_at_n_scores_the_same_text_as_its_siblings():
    doc, response = _doc(), _response_with_reasoning(["Paris"] * 4)

    assert MajAtN(n=4).compute(doc=doc, model_response=response) == 1
    assert PassAtK(k=1, n=4).compute(doc=doc, model_response=response) == 1.0
    assert AvgAtN(n=4).compute(doc=doc, model_response=response) == 1.0


@pytest.mark.parametrize(
    "answers, expected",
    [
        ([" Paris", "Paris\n", " Paris ", "Paris"], 1.0),
        ([" Paris", " London", "Paris ", "Paris"], 0.75),
        (["London", "London", "London", "London"], 0.0),
    ],
    ids=["all-correct-with-whitespace", "mixed", "all-wrong-legitimate-zero"],
)
def test_avg_at_n_applies_strip_strings(answers, expected):
    """The last case is the legitimate zero: it must keep returning 0.0."""
    response = ModelResponse(text=list(answers))
    assert AvgAtN(n=4, strip_strings=True).compute(doc=_doc(), model_response=response) == expected


def test_avg_at_n_raises_when_n_unset():
    """Slicing with an unset n used to raise TypeError; it must stay loud."""
    with pytest.raises(Exception, match="You did not set the value of n"):
        AvgAtN().compute(doc=_doc(), model_response=ModelResponse(text=["Paris"]))

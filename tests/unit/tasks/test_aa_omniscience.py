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
from inspect_ai.scorer import SampleScore, Target, mean_score

from lighteval.tasks.tasks import aa_omniscience


class FakeGrader:
    def __init__(self, replies):
        self.replies = iter(replies)

    async def generate(self, prompt):
        return SimpleNamespace(completion=next(self.replies))


def grade_samples(grader_replies, monkeypatch):
    """Score one sample per grader reply, then reduce each score the way inspect_ai does for a single epoch."""
    grader = FakeGrader(grader_replies)
    monkeypatch.setattr(aa_omniscience, "get_model", lambda *args, **kwargs: grader)
    state = SimpleNamespace(input_text="question", output=SimpleNamespace(completion="answer"))

    scores = []
    for sample_id in range(len(grader_replies)):
        score = asyncio.run(aa_omniscience.aa_omniscience_scorer()(state, Target("answer")))
        scores.append(SampleScore(score=mean_score()([score]), sample_id=sample_id))
    return scores


@pytest.mark.parametrize(
    "grader_replies, omniscience_index, accuracy, hallucination_rate",
    [
        ("AAAABBCCDD", 0.2, 0.4, 2 / 6),
        ("AAAADDDDDD", 0.4, 0.4, 0.0),
        ("AAAABBBBBB", -0.2, 0.4, 1.0),
        ("AAAA", 1.0, 1.0, 0.0),
        (["A\n", "B"], 0.0, 0.5, 1.0),
        (["A", "B", "B", "Grade: B"], -0.25, 0.25, 2 / 3),
    ],
    ids=[
        "mixed",
        "abstains_when_unsure",
        "guesses_when_unsure",
        "all_correct",
        "grade_with_trailing_newline",
        "unparsed_grade_counts_as_not_attempted",
    ],
)
def test_metrics_follow_paper_definitions(
    grader_replies, omniscience_index, accuracy, hallucination_rate, monkeypatch
):
    scores = grade_samples(grader_replies, monkeypatch)

    assert aa_omniscience.omniscience_index()(scores) == pytest.approx(omniscience_index)
    assert aa_omniscience.accuracy()(scores) == pytest.approx(accuracy)
    assert aa_omniscience.hallucination_rate()(scores) == pytest.approx(hallucination_rate)

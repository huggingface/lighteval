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

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.multilingual.tasks.swiss_legal.metrics import (
    JudgeSwissLandmarkDecisionSummarization,
    JudgeSwissLegalTranslation,
)
from lighteval.tasks.requests import Doc


class RecordingJudge:
    """Stands in for the LLM judge and records the answers it is asked to grade."""

    def __init__(self):
        self.answers = None

    def evaluate_answer_batch(self, questions, answers, options, golds):
        self.answers = list(answers)
        return [1.0] * len(answers), [None] * len(answers), ["ok"] * len(answers)


def build_metric(metric_class):
    metric = object.__new__(metric_class)  # skip the judge backend setup, which needs a model or an API key
    metric.short_judge_name = "judge"
    metric.judge = RecordingJudge()
    return metric


def reasoning_response(answer):
    # What the pipeline stores for a reasoning model once reasoning tags are removed
    return ModelResponse(text=[f"<think>The user wants a translation.</think>{answer}"], text_post_processed=[answer])


@pytest.mark.parametrize("metric_class", [JudgeSwissLegalTranslation, JudgeSwissLandmarkDecisionSummarization])
def test_judges_grade_the_text_without_reasoning(metric_class):
    metric = build_metric(metric_class)
    doc = Doc(query="Translate", choices=["Guten Tag"], gold_index=0, specific={"source": "Bonjour"})

    results = metric.compute(responses=[reasoning_response("Guten Tag")], docs=[doc])

    assert metric.judge.answers == ["Guten Tag"]
    assert results == [{"judge": 100.0}]


@pytest.mark.parametrize("metric_class", [JudgeSwissLegalTranslation, JudgeSwissLandmarkDecisionSummarization])
def test_judges_grade_the_raw_text_when_reasoning_is_kept(metric_class):
    metric = build_metric(metric_class)
    doc = Doc(query="Translate", choices=["Guten Tag"], gold_index=0, specific={"source": "Bonjour"})

    metric.compute(responses=[ModelResponse(text=["<think>kept</think>Guten Tag"])], docs=[doc])

    assert metric.judge.answers == ["<think>kept</think>Guten Tag"]

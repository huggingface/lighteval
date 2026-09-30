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

import asyncio

import pytest
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks.bigbench_extra_hard import (
    TASKS_TABLE,
    bbeh_inspect_scorer,
    bbeh_metric,
    bbeh_prompt,
    evaluate_bbeh_correctness,
)


def test_bbeh_prompt_full_gold_preservation():
    """Verify that Doc.choices contains the full target string rather than individual characters."""
    line = {
        "input": "dummy question",
        "target": "positive",
    }
    doc = bbeh_prompt(line, task_name="bigbench_extra_hard:test")
    assert doc.choices == ["positive"]
    assert doc.get_golds() == ["positive"]
    assert doc.gold_index == 0


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("4", "4", True),
        ("The final answer is: \\boxed{4}.", "4", True),
        ("The answer is: (A)", "a", True),
        ("25", "25.0", True),
        ("2, 3, 4", "2,3,4", True),
        ("[foo]", "foo", True),
        ("foo?", "foo", True),
        ("'foo'", "foo", True),
        ("\\boxed{4}", "4", True),
        ("$4$", "4", True),
        ("\\text{foo}", "foo", True),
        ("\\texttt{foo}", "foo", True),
        ("Ok The answer is: **25**\nHere's why.", "25.0", True),
        ("The answer is: 5", "4", False),
        ("bar", "foo", False),
        ("I don't know the answer.", "4", False),
    ],
)
def test_bbeh_answer_matching(prediction: str, reference: str, expected: bool):
    """Verify BBEH normalization, extraction, and fuzzy equivalence matching."""
    assert evaluate_bbeh_correctness(prediction, reference) is expected


def test_bbeh_native_metric_integration():
    """Verify that the BBEH SampleLevelMetric computes accuracy on ModelResponse against Doc."""
    line = {
        "input": "What is 2 + 2?",
        "target": "4",
    }
    doc = bbeh_prompt(line, task_name="bigbench_extra_hard:test")

    # Correct response
    correct_response = ModelResponse(text=["The final answer is: \\boxed{4}."])
    result_correct = bbeh_metric.compute_sample(doc=doc, model_response=correct_response)
    assert result_correct == {"acc": 1.0}

    # Incorrect response
    incorrect_response = ModelResponse(text=["The answer is: 5"])
    result_incorrect = bbeh_metric.compute_sample(doc=doc, model_response=incorrect_response)
    assert result_incorrect == {"acc": 0.0}


def test_bbeh_sampling_category_and_table_wiring():
    """Guard against reverting BBEH to log-probabilities or misconfiguring tasks in TASKS_TABLE."""
    assert bbeh_metric.category == SamplingMethod.GENERATIVE
    assert bbeh_metric.metric_name == "acc"

    assert len(TASKS_TABLE) == 23
    for task in TASKS_TABLE:
        assert len(task.metrics) == 1
        assert task.metrics[0].category == SamplingMethod.GENERATIVE
        assert task.metrics[0].metric_name == "acc"


def test_bbeh_inspect_scorer():
    """Verify that the Inspect scorer extracts answers and applies BBEH fuzzy equivalence."""
    scorer = bbeh_inspect_scorer()

    async def run_inspect_tests():
        # Matching fuzzy case
        state_fuzzy = TaskState(
            model="test",
            sample_id=1,
            epoch=1,
            input="Question: ...?\nAnswer:",
            messages=[ChatMessageAssistant(content="ANSWER: (A)")],
            output=ModelOutput.from_content("test", "ANSWER: (A)"),
        )
        score_fuzzy = await scorer(state_fuzzy, Target("a"))
        assert score_fuzzy.value == "C"
        assert score_fuzzy.answer == "(A)"

        # Mismatch case
        state_mismatch = TaskState(
            model="test",
            sample_id=2,
            epoch=1,
            input="Question: ...?\nAnswer:",
            messages=[ChatMessageAssistant(content="ANSWER: (B)")],
            output=ModelOutput.from_content("test", "ANSWER: (B)"),
        )
        score_mismatch = await scorer(state_mismatch, Target("a"))
        assert score_mismatch.value == "I"
        assert score_mismatch.answer == "(B)"

        # Extraction failure case (no ANSWER prefix)
        state_no_ans = TaskState(
            model="test",
            sample_id=3,
            epoch=1,
            input="Question: ...?\nAnswer:",
            messages=[ChatMessageAssistant(content="I do not know")],
            output=ModelOutput.from_content("test", "I do not know"),
        )
        score_no_ans = await scorer(state_no_ans, Target("a"))
        assert score_no_ans.value == "N"
        assert score_no_ans.answer is None

    asyncio.run(run_inspect_tests())

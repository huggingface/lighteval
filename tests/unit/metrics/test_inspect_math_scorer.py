# MIT License

# Copyright (c) 2026 The HuggingFace Team

"""Regression coverage for mathematical equivalence in the Inspect scorer."""

import asyncio

import pytest
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from lighteval.metrics.metrics import math_scorer


@pytest.mark.parametrize(
    "gold,prediction,expected",
    [
        ("42", r"\boxed{42}", "C"),
        ("42", r"\boxed{42.0}", "C"),
        ("42", r"\boxed{042}", "C"),
        ("1000", r"\boxed{1,000}", "C"),
        ("0.5", r"\boxed{0.50}", "C"),
        ("0.5", r"\boxed{\frac{1}{2}}", "C"),
        ("2", r"\boxed{1+1}", "C"),
        ("42", r"\boxed{41}", "I"),
        ("42", "I do not know", "I"),
        ("abc", "I do not know", "I"),
        ("0.5", r"\boxed{\frac{2}{3}}", "I"),
        ("42", "", "I"),
    ],
)
def test_math_scorer_uses_mathematical_equivalence(gold, prediction, expected):
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="test",
        epoch=1,
        input="",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content=prediction),
    )
    result = asyncio.run(math_scorer()(state, Target(gold)))
    assert result.value == expected

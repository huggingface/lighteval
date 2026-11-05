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

import pytest
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from lighteval.metrics.metrics import math_scorer


class TestInspectMathScorer:
    """Tests for inspect-ai math_scorer() handling mathematical equivalence and false-positive guards."""

    @pytest.fixture
    def scorer(self):
        return math_scorer()

    def _score(self, scorer, target: str, completion: str) -> str:
        state = TaskState(
            model=ModelName("mockllm/model"),
            sample_id="test",
            epoch=1,
            input="",
            messages=[],
            output=ModelOutput.from_content(model="mockllm/model", content=completion),
        )
        result = asyncio.run(scorer(state, Target(target)))
        return result.value

    @pytest.mark.parametrize(
        ("target", "completion"),
        [
            # Canonical boxed answers
            ("42", r"\boxed{42}"),
            # Leading zeros (AIME answer format)
            ("42", r"\boxed{042}"),
            # Decimals / Floats
            ("42", r"\boxed{42.0}"),
            # Thousands separators (commas)
            ("1000", r"\boxed{1,000}"),
            # Negative numbers
            ("-15", r"\boxed{-15}"),
            ("-15", r"\boxed{-15.0}"),
            # Decimal to LaTeX fraction
            ("0.5", r"\boxed{\frac{1}{2}}"),
            ("1.5", r"\boxed{\frac{3}{2}}"),
            # Fraction target to decimal
            ("1/2", r"\boxed{0.5}"),
            ("1/4", r"\boxed{0.25}"),
            # Arithmetic expressions
            ("2", r"\boxed{1+1}"),
            ("12", r"\boxed{3 \times 4}"),
            # Chain-of-thought completion with boxed answer
            ("42", r"We calculate 40 + 2 = 42. So the answer is \boxed{42}."),
            ("1000", r"Total cost: 10 * 100 = 1000. Hence, \boxed{1,000}."),
            # GSM8K hash-style output delimiter
            ("42", "The answer is #### 42"),
        ],
    )
    def test_math_scorer_equivalent_answers(self, scorer, target, completion):
        assert self._score(scorer, target, completion) == "C"

    @pytest.mark.parametrize(
        ("target", "completion"),
        [
            # Incorrect values
            ("42", r"\boxed{43}"),
            # Wrong sign
            ("42", r"\boxed{-42}"),
            # Off by one
            ("1000", r"\boxed{1001}"),
            # Mismatched fraction
            ("0.5", r"\boxed{\frac{1}{3}}"),
        ],
    )
    def test_math_scorer_incorrect_answers(self, scorer, target, completion):
        assert self._score(scorer, target, completion) == "I"

    @pytest.mark.parametrize(
        ("target", "completion"),
        [
            # Issue #1430 false-positive regression: [] == [] should NOT be marked correct
            ("abc", "I do not know"),
            # Model refusal
            ("42", "I am unable to answer this question."),
            # Empty completion
            ("42", ""),
            # Empty target and empty completion
            ("", ""),
        ],
    )
    def test_math_scorer_empty_and_unparseable_guards(self, scorer, target, completion):
        assert self._score(scorer, target, completion) == "I"

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
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState

from lighteval.tasks.tasks.dyck_language import dyck_exact


def score_answer(answer: str, target: str) -> str:
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id=0,
        epoch=1,
        input="",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content=answer),
    )
    return asyncio.run(dyck_exact()(state, Target(target)))


@pytest.mark.parametrize(
    "answer, target, expected",
    [
        (" ] ) ]", " ] ) ]", CORRECT),
        ("])]", " ] ) ]", CORRECT),  # whitespace is not part of the answer
        (" ] ] )", " ] ) ]", INCORRECT),  # right bracket types, wrong order
        (" ) ) )", " ] ) ]", INCORRECT),  # wrong bracket types
        (" ] )", " ] ) ]", INCORRECT),  # missing a bracket
        ("", " ] ) ]", INCORRECT),
    ],
)
def test_dyck_exact(answer: str, target: str, expected: str):
    assert score_answer(answer, target).value == expected

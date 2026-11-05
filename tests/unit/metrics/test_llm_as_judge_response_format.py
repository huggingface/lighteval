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

"""Regression tests for JudgeLM response_format defaulting."""

from pydantic import BaseModel

from lighteval.metrics.utils.llm_as_judge import DEFAULT_FORMAT, JudgeLM


def _make_judge(**kwargs):
    return JudgeLM(
        model="gpt-4o",
        templates=lambda **kw: [{"role": "user", "content": "q"}],
        process_judge_response=lambda x: x,
        judge_backend="openai",
        **kwargs,
    )


def test_response_format_defaults_when_not_provided():
    """When response_format is not given, DEFAULT_FORMAT must be applied.

    The previous expression `response_format if not None else DEFAULT_FORMAT`
    tested the constant `not None` (always True), so DEFAULT_FORMAT was
    unreachable and None leaked into the request kwargs of every backend.
    """
    judge = _make_judge()

    assert judge.response_format == DEFAULT_FORMAT


def test_explicit_response_format_is_preserved():
    class JudgeVerdict(BaseModel):
        score: float

    judge = _make_judge(response_format=JudgeVerdict)

    assert judge.response_format is JudgeVerdict

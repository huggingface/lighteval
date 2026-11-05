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

from pydantic import BaseModel

from lighteval.metrics.utils.llm_as_judge import DEFAULT_FORMAT, JudgeLM


class Verdict(BaseModel):
    score: int


def _make_judge(**kwargs):
    return JudgeLM(
        model="openai/dummy-model",
        templates=lambda question, answer, options=None, gold=None, **kw: [{"role": "user", "content": question}],
        process_judge_response=lambda response: response,
        judge_backend="litellm",
        **kwargs,
    )


def test_response_format_falls_back_to_default():
    """A judge built without an explicit `response_format` gets `DEFAULT_FORMAT`."""
    assert _make_judge().response_format == DEFAULT_FORMAT


def test_explicit_response_format_is_kept():
    """An explicit `response_format` is never overwritten by the default."""
    assert _make_judge(response_format={"type": "json_object"}).response_format == {"type": "json_object"}
    assert _make_judge(response_format=Verdict).response_format is Verdict

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

"""Regression tests for the litellm judge backend request construction (issue #1296)."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from lighteval.metrics.utils.llm_as_judge import JudgeLM


def _stub_litellm(captured_calls: list) -> types.ModuleType:
    """A minimal litellm stand-in that records completion kwargs."""
    stub = types.ModuleType("litellm")
    stub.drop_params = False
    stub.supports_reasoning = lambda model: False

    def completion(**kwargs):
        captured_calls.append(kwargs)
        message = MagicMock()
        message.content = "judge says ok"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    stub.completion = completion
    return stub


@pytest.fixture
def litellm_calls(monkeypatch):
    captured: list = []
    monkeypatch.setitem(sys.modules, "litellm", _stub_litellm(captured))
    return captured


def _make_judge(max_tokens):
    return JudgeLM(
        model="gpt-4o",
        templates=lambda **kwargs: [{"role": "user", "content": "q"}],
        process_judge_response=lambda x: x,
        judge_backend="litellm",
        max_tokens=max_tokens,
        backend_options={"caching": False, "increase_max_tokens_for_reasoning": False},
    )


def test_litellm_max_tokens_is_sent_as_integer(litellm_calls):
    """max_tokens must reach the API as an int, not a 1-tuple (issue #1296).

    A tuple serializes to a JSON array ("max_tokens": [512]) that
    OpenAI-compatible servers reject with a 400; after the retries the judge
    would silently score the error string instead of a real response.
    """
    judge = _make_judge(max_tokens=512)

    results = judge._JudgeLM__call_litellm([[{"role": "user", "content": "q"}]])

    assert results == ["judge says ok"]
    assert litellm_calls, "litellm.completion was never called"
    max_tokens = litellm_calls[0]["max_tokens"]
    assert isinstance(max_tokens, int)
    assert max_tokens == 512


def test_litellm_omits_max_tokens_when_unset(litellm_calls):
    judge = _make_judge(max_tokens=None)

    judge._JudgeLM__call_litellm([[{"role": "user", "content": "q"}]])

    assert "max_tokens" not in litellm_calls[0]

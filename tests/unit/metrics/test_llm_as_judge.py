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

import sys
from types import ModuleType, SimpleNamespace

import pytest

from lighteval.metrics.utils.llm_as_judge import JudgeLM


def _fake_completion_response(text: str = "judgment"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _build_fake_litellm(captured: dict) -> ModuleType:
    """Return a minimal fake litellm module that records kwargs passed to completion.

    __call_litellm does `import litellm` inside the function body (not at module
    scope), so injecting into sys.modules is enough for the import to resolve against
    this stub without litellm being installed.

    Attributes provided:
    - completion: records kwargs and returns a stub response
    - supports_reasoning: always False (non-reasoning model; prevents the
      max_tokens * 10 rewrite so tests assert on the value the caller passed in)
    - drop_params: writable attribute set by __call_litellm before each call
    """
    fake = ModuleType("litellm")

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_completion_response()

    fake.completion = fake_completion
    fake.supports_reasoning = lambda model: False
    return fake


def _make_judge(**kwargs):
    return JudgeLM(
        model="openai/dummy-model",
        templates=lambda question, answer, options=None, gold=None, **kw: [{"role": "user", "content": question}],
        process_judge_response=lambda response: response,
        judge_backend="litellm",
        # `increase_max_tokens_for_reasoning` would rewrite max_tokens to
        # min(max_tokens * 10, 32000) for reasoning models; disable it so the
        # test asserts on the value the caller passed in, not the rewritten one.
        backend_options={"caching": False, "increase_max_tokens_for_reasoning": False},
        **kwargs,
    )


@pytest.mark.parametrize("max_tokens", [64, 512])
def test_litellm_judge_passes_max_tokens_as_int(monkeypatch, max_tokens):
    """`max_tokens` must reach litellm as an int, not a one-element sequence.

    Sending a sequence makes any spec-compliant OpenAI-compatible server reject the
    request with a 400, after which the judge returns its error string and that string
    gets scored as a real judgment.

    The parametrization over two values guards against a fix that hard-codes a
    single value instead of passing through whatever the caller set.
    """
    # Inject the stub before the judge is called. __call_litellm does
    # `import litellm` at call time, so sys.modules patching is enough;
    # no real litellm install is needed and the test will not be skipped in CI.
    captured = {}
    monkeypatch.setitem(sys.modules, "litellm", _build_fake_litellm(captured))

    judge = _make_judge(max_tokens=max_tokens)
    # evaluate_answer_batch is the public entry point; calling it also exercises
    # the "the judge returned a real judgment and not an error string" path,
    # since responses == ["judgment"] would be ["ERROR: ..."] if the call failed.
    _, _, responses = judge.evaluate_answer_batch(
        questions=["Why is 2 + 2 = 4?"], answers=["4"], options=[None], golds=["4"]
    )

    assert captured["max_tokens"] == max_tokens
    # isinstance documents the intent: a 1-tuple (max_tokens,) compares unequal
    # to max_tokens, so == alone would already catch the regression, but the
    # type check makes the failure message immediately obvious if it regresses.
    assert isinstance(captured["max_tokens"], int)
    assert responses == ["judgment"]


def test_litellm_judge_omits_max_tokens_when_unset(monkeypatch):
    """No `max_tokens` on the judge means no `max_tokens` key in the litellm call.

    Guards against the obvious wrong fix of always sending max_tokens=None,
    which some providers reject or handle differently from omitting the key.
    """
    # Same stub approach as the parametrized test above.
    captured = {}
    monkeypatch.setitem(sys.modules, "litellm", _build_fake_litellm(captured))

    judge = _make_judge()
    judge.evaluate_answer_batch(questions=["Why is 2 + 2 = 4?"], answers=["4"], options=[None], golds=["4"])

    assert "max_tokens" not in captured

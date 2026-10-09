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
import types

from lighteval.metrics.utils.llm_as_judge import JudgeLM


def build_fake_litellm(calls: list[dict]):
    """Minimal stand-in for the litellm module.

    litellm is not part of the base test dependencies, so rather than install it we
    register a fake module. `JudgeLM.__call_litellm` does `import litellm` lazily, so
    it resolves to this. `completion` simply records the kwargs it was called with,
    which is what we assert against.
    """

    module = types.ModuleType("litellm")
    module.drop_params = False

    def completion(**kwargs):
        calls.append(kwargs)
        message = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    module.completion = completion
    # never take the reasoning-model branch, so max_tokens is left as configured
    module.supports_reasoning = lambda model: False
    return module


def build_judge(max_tokens):
    return JudgeLM(
        model="gpt-4o-mini",
        templates=lambda **kwargs: [],
        process_judge_response=lambda response: response,
        judge_backend="litellm",
        max_tokens=max_tokens,
        # caching off so __call_litellm doesn't import litellm.caching
        backend_options={"caching": False},
    )


class TestLiteLLMJudgeMaxTokens:
    def test_max_tokens_is_sent_as_int(self, monkeypatch):
        """max_tokens must reach litellm as an int, not a 1-tuple.

        A trailing comma previously made this `(max_tokens,)`, which litellm serializes
        to the JSON array `[512]`. Spec-compliant OpenAI-compatible servers reject that
        with 400, and JudgeLM then scores the resulting error message.
        """

        calls: list[dict] = []
        monkeypatch.setitem(sys.modules, "litellm", build_fake_litellm(calls))

        judge = build_judge(max_tokens=512)
        judge._JudgeLM__call_litellm([[{"role": "user", "content": "hello"}]])

        assert len(calls) == 1
        assert calls[0]["max_tokens"] == 512
        assert isinstance(calls[0]["max_tokens"], int)

    def test_max_tokens_omitted_when_unset(self, monkeypatch):
        """When max_tokens is None, the key must not be sent at all."""

        calls: list[dict] = []
        monkeypatch.setitem(sys.modules, "litellm", build_fake_litellm(calls))

        judge = build_judge(max_tokens=None)
        judge._JudgeLM__call_litellm([[{"role": "user", "content": "hello"}]])

        assert len(calls) == 1
        assert "max_tokens" not in calls[0]

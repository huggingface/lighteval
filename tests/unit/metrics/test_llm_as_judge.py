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

from lighteval.metrics.utils.llm_as_judge import JudgeLM


def _fake_completion_response(content):
    """Minimal stand-in for a litellm ModelResponse."""

    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    return _Response(content)


def _make_judge():
    return JudgeLM(
        model="gpt-4o",
        templates=lambda question, answer, options=None, gold=None, **kwargs: [{"role": "user", "content": answer}],
        # Mirrors process_judge_response_simpleqa: anything unrecognised scores 0.0.
        process_judge_response=lambda response: 1.0 if response == "A" else 0.0,
        judge_backend="litellm",
    )


@pytest.fixture
def no_judge_caching(monkeypatch):
    """Disk caching would need litellm[caching] and touch the filesystem."""
    import litellm

    monkeypatch.setattr(litellm, "supports_reasoning", lambda *args, **kwargs: False)
    judge = _make_judge()
    judge.backend_options.caching = False
    return judge, litellm


def test_litellm_judge_raises_when_api_never_responds(no_judge_caching, monkeypatch):
    """An unreachable judge API must be surfaced, not scored.

    Regression test: the litellm backend used to return the sentinel string
    "ERROR: Failed to get response from the API." after exhausting its retries.
    That string was passed to process_judge_response, which has no branch for it
    and returns 0.0 -- so an API outage was silently recorded as the evaluated
    model answering incorrectly, quietly depressing the benchmark score.
    """
    judge, litellm = no_judge_caching
    calls = []

    def always_fails(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("429 RateLimitError: judge API over quota")

    monkeypatch.setattr(litellm, "completion", always_fails)
    monkeypatch.setattr(judge, "API_RETRY_SLEEP", 0)

    with pytest.raises(ValueError, match="not annotated"):
        judge.evaluate_answer_batch(
            questions=["What is the capital of France?"],
            answers=["Paris"],
            options=[None],
            golds=["Paris"],
        )

    assert len(calls) == judge.API_MAX_RETRY


def test_litellm_judge_scores_normally_when_api_responds(no_judge_caching, monkeypatch):
    """The healthy path is unchanged by the failure handling above."""
    judge, litellm = no_judge_caching

    monkeypatch.setattr(litellm, "completion", lambda **kwargs: _fake_completion_response("A"))

    scores, _prompts, responses = judge.evaluate_answer_batch(
        questions=["What is the capital of France?"],
        answers=["Paris"],
        options=[None],
        golds=["Paris"],
    )

    assert responses == ["A"]
    assert scores == [1.0]

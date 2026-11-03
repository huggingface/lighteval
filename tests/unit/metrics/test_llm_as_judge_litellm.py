# MIT License
#
# Copyright (c) 2024 The HuggingFace Team

from types import SimpleNamespace

from lighteval.metrics.utils.llm_as_judge import JudgeLM


class _FakeLiteLLM:
    drop_params = False
    cache = None

    def __init__(self):
        self.max_tokens = None

    @staticmethod
    def supports_reasoning(model):
        return False

    def completion(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens")
        message = SimpleNamespace(content="1")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_litellm_judge_sends_max_tokens_as_int(monkeypatch):
    fake = _FakeLiteLLM()
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)

    judge = JudgeLM(
        model="openai/test-model",
        templates=lambda question, answer, options=None, gold=None, **kw: [
            {"role": "user", "content": question}
        ],
        process_judge_response=lambda response: response,
        judge_backend="litellm",
        max_tokens=64,
        backend_options={"caching": False, "increase_max_tokens_for_reasoning": False},
    )

    scores, _, responses = judge.evaluate_answer_batch(["q"], ["a"], [None], [None])

    assert fake.max_tokens == 64
    assert isinstance(fake.max_tokens, int)
    assert scores == ["1"]
    assert responses == ["1"]

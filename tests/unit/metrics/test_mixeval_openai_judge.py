import json

import httpx
import pytest
from openai import OpenAI

from lighteval.metrics.utils.llm_as_judge import JudgeLM
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.mix_eval.main import (
    llm_judge_mixeval_freeform_gpt_judge,
    llm_judge_mixeval_multichoice_gpt_judge,
)


def mock_client(requests: list, answer: str) -> OpenAI:
    def handler(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": answer}
        return httpx.Response(
            200,
            json={
                "id": "id",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            },
        )

    return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize(
    "metric, answer",
    [(llm_judge_mixeval_multichoice_gpt_judge, "[[1]]"), (llm_judge_mixeval_freeform_gpt_judge, "[[1.0]]")],
)
def test_mixeval_gpt_judge_request(monkeypatch, metric, answer):
    requests = []
    judge = metric.sample_level_fn.judge
    monkeypatch.setattr(judge, "client", mock_client(requests, answer))
    doc = Doc(query="q", choices=["a", "b"], gold_index=0, specific={"question": "q"})

    result = metric.sample_level_fn.compute(responses=[ModelResponse(text=["a"])], docs=[doc])

    assert requests[0]["model"] == "gpt-5.6-terra"
    assert "max_tokens" not in requests[0]
    assert "temperature" not in requests[0]
    assert result == [{"judge_score_gpt-5.6-terra": 1}]
    assert set(result[0]) == set(metric.corpus_level_fn) == set(metric.higher_is_better)


@pytest.mark.parametrize(
    "model, token_param",
    [("gpt-5.6-terra", "max_completion_tokens"), ("gpt-4o", "max_tokens")],
)
def test_openai_judge_token_limit_param(model, token_param):
    requests = []
    judge = JudgeLM(
        model=model,
        templates=lambda **kwargs: [{"role": "user", "content": "q"}],
        process_judge_response=lambda x: x,
        judge_backend="openai",
        max_tokens=16,
    )
    judge.client = mock_client(requests, "ok")

    judge.evaluate_answer("q", "a")

    assert requests[0][token_param] == 16
    assert len({"max_tokens", "max_completion_tokens"} & set(requests[0])) == 1

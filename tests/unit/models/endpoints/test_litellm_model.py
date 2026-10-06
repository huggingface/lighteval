# MIT License

# Copyright (c) 2026 The HuggingFace Team

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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("litellm")

from lighteval.models.endpoints.litellm_model import LiteLLMClient, LiteLLMModelConfig, litellm


def make_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_client(cache_dir: str, api_max_retry: int = 3) -> LiteLLMClient:
    return LiteLLMClient(
        LiteLLMModelConfig(
            model_name="openai/gpt-4o-mini",
            provider="openai",
            api_key="test-key",
            api_max_retry=api_max_retry,
            cache_dir=cache_dir,
        )
    )


class TestLiteLLMClientRetries:
    def test_rate_limit_error_still_retries(self, monkeypatch, tmp_path):
        client = make_client(str(tmp_path), api_max_retry=3)
        calls = []
        responses = [
            litellm.RateLimitError(
                message="rate limited",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            ),
            litellm.RateLimitError(
                message="rate limited",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            ),
            make_response("ok"),
        ]

        def fake_completion(**kwargs):
            calls.append(kwargs)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr("lighteval.models.endpoints.litellm_model.litellm.completion", fake_completion)
        sleep_calls = []
        monkeypatch.setattr("lighteval.models.endpoints.litellm_model.time.sleep", sleep_calls.append)

        response = client._LiteLLMClient__call_api(
            prompt=[{"role": "user", "content": "hi"}],
            return_logits=False,
            max_new_tokens=10,
            num_samples=1,
            stop_sequence=None,
        )

        assert response.choices[0].message.content == "ok"
        assert len(calls) == 3
        assert sleep_calls == [1.0, 2.0]

    def test_non_retriable_status_code_fails_fast(self, monkeypatch, tmp_path):
        client = make_client(str(tmp_path))
        calls = []

        class FakeException(Exception):
            status_code = 401

        def fake_completion(**kwargs):
            calls.append(kwargs)
            raise FakeException("unauthorized")

        monkeypatch.setattr("lighteval.models.endpoints.litellm_model.litellm.completion", fake_completion)
        sleep = Mock()
        monkeypatch.setattr("lighteval.models.endpoints.litellm_model.time.sleep", sleep)

        with pytest.raises(FakeException):
            client._LiteLLMClient__call_api(
                prompt=[{"role": "user", "content": "hi"}],
                return_logits=False,
                max_new_tokens=10,
                num_samples=1,
                stop_sequence=None,
            )

        assert len(calls) == 1
        sleep.assert_not_called()

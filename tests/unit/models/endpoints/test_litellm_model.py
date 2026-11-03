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

import logging
from unittest.mock import patch

import litellm
import pytest

from lighteval.models.endpoints.litellm_model import LiteLLMClient


COMPLETION = "lighteval.models.endpoints.litellm_model.litellm.completion"
SLEEP = "lighteval.models.endpoints.litellm_model.time.sleep"


def _client():
    # Bypass __init__: only the retry settings are needed by __call_api.
    client = LiteLLMClient.__new__(LiteLLMClient)
    client.API_MAX_RETRY = 3
    client.API_RETRY_SLEEP = 1
    client.API_RETRY_MULTIPLIER = 2
    client.model = "gpt-4o"
    client.provider = "openai"
    client.base_url = None
    client.api_key = None
    client.timeout = None

    class _Params:
        def to_litellm_dict(self):
            return {}

    client.generation_parameters = _Params()
    return client


def _call(client):
    return client._LiteLLMClient__call_api([{"role": "user", "content": "hi"}], False, 8, 1, None)


def _bad_request(message):
    return litellm.BadRequestError(message=message, model="gpt-4o", llm_provider="openai")


@pytest.mark.parametrize("message", ["Unsupported parameter: 'logprobs'"])
def test_bad_request_is_logged_and_backed_off(message, caplog):
    client = _client()
    with (
        patch(COMPLETION, side_effect=_bad_request(message)) as comp,
        patch(SLEEP) as sleep,
        caplog.at_level(logging.WARNING),
    ):
        _call(client)
    assert comp.call_count == client.API_MAX_RETRY
    assert sleep.call_count == client.API_MAX_RETRY
    assert "Unsupported parameter" in caplog.text


def test_content_filter_bad_request_returns_empty_without_retry():
    client = _client()
    message = "The response was filtered due to the prompt triggering Microsoft's content management policy"
    with (
        patch(COMPLETION, side_effect=_bad_request(message)) as comp,
        patch(SLEEP) as sleep,
    ):
        _call(client)
    assert comp.call_count == 1
    sleep.assert_not_called()

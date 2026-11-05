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

from unittest.mock import Mock, patch

import pytest

from lighteval.models.endpoints.litellm_model import LiteLLMClient
from lighteval.models.model_input import GenerationParameters
from lighteval.utils.imports import is_package_available


pytestmark = pytest.mark.skipif(not is_package_available("litellm"), reason="litellm extra is not installed")


def _build_client(model_name: str, generation_parameters: GenerationParameters) -> LiteLLMClient:
    client = LiteLLMClient.__new__(LiteLLMClient)
    client.model = model_name
    client.provider = "openai"
    client.base_url = None
    client.api_key = None
    client.generation_parameters = generation_parameters
    client._max_length = 10_000
    client.API_MAX_RETRY = 1
    client.API_RETRY_SLEEP = 0
    client.API_RETRY_MULTIPLIER = 1
    client.timeout = None
    return client


@pytest.mark.parametrize(
    "reasoning_effort, supports_reasoning_value, expected_prepared_max_new_tokens",
    [
        (None, True, 100),
        ("none", True, 100),
        ("low", False, 100),
        ("low", True, 1000),
    ],
)
def test_prepare_max_new_tokens_boosts_only_with_reasoning_effort(
    reasoning_effort: str | None, supports_reasoning_value: bool, expected_prepared_max_new_tokens: int
):
    client = _build_client("openai/o3-mini", GenerationParameters(reasoning_effort=reasoning_effort))

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=supports_reasoning_value):
        assert client._prepare_max_new_tokens(100) == expected_prepared_max_new_tokens


def test_call_api_o_series_keeps_reasoning_effort_but_drops_sampling_params():
    client = _build_client("openai/o3-mini", GenerationParameters(temperature=0.2, top_p=0.9, reasoning_effort="low"))
    response = Mock()
    response.choices = [Mock(message=Mock(content="ok"))]

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.completion", return_value=response) as completion:
            client._LiteLLMClient__call_api(
                prompt=[{"role": "user", "content": "hello"}],
                return_logits=False,
                max_new_tokens=64,
                num_samples=1,
                stop_sequence=None,
            )

    completion_kwargs = completion.call_args.kwargs
    assert completion_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in completion_kwargs
    assert "top_p" not in completion_kwargs


def test_call_api_non_o_series_passes_full_litellm_generation_kwargs():
    client = _build_client(
        "google/gemini-2.5-flash", GenerationParameters(temperature=0.2, top_p=0.9, reasoning_effort="low")
    )
    response = Mock()
    response.choices = [Mock(message=Mock(content="ok"))]

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.completion", return_value=response) as completion:
            client._LiteLLMClient__call_api(
                prompt=[{"role": "user", "content": "hello"}],
                return_logits=False,
                max_new_tokens=64,
                num_samples=1,
                stop_sequence=None,
            )

    completion_kwargs = completion.call_args.kwargs
    assert completion_kwargs["temperature"] == 0.2
    assert completion_kwargs["top_p"] == 0.9
    assert completion_kwargs["reasoning_effort"] == "low"


def test_call_api_openai_non_reasoning_uses_only_max_tokens():
    client = _build_client("openai/gpt-4.1-nano", GenerationParameters(max_new_tokens=96))
    response = Mock()
    response.choices = [Mock(message=Mock(content="ok"))]

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.completion", return_value=response) as completion:
            client._LiteLLMClient__call_api(
                prompt=[{"role": "user", "content": "hello"}],
                return_logits=False,
                max_new_tokens=64,
                num_samples=1,
                stop_sequence=None,
            )

    completion_kwargs = completion.call_args.kwargs
    assert completion_kwargs["max_tokens"] == 64
    assert "max_completion_tokens" not in completion_kwargs


def test_call_api_openai_reasoning_keeps_max_completion_tokens():
    client = _build_client("openai/gpt-5-mini", GenerationParameters(max_new_tokens=96, reasoning_effort="low"))
    response = Mock()
    response.choices = [Mock(message=Mock(content="ok"))]

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=True):
        with patch("lighteval.models.endpoints.litellm_model.litellm.completion", return_value=response) as completion:
            client._LiteLLMClient__call_api(
                prompt=[{"role": "user", "content": "hello"}],
                return_logits=False,
                max_new_tokens=64,
                num_samples=1,
                stop_sequence=None,
            )

    completion_kwargs = completion.call_args.kwargs
    assert completion_kwargs["max_tokens"] == 640
    assert completion_kwargs["max_completion_tokens"] == 96

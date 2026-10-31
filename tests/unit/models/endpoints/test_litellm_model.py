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

from types import SimpleNamespace

from lighteval.models.endpoints import litellm_model


def test_litellm_config_accepts_cache_dir():
    config = litellm_model.LiteLLMModelConfig(model_name="openai/test", litellm_cache_dir="/tmp/litellm")
    assert config.litellm_cache_dir == "/tmp/litellm"


def test_litellm_cache_dir_prefers_config_over_environment(monkeypatch):
    monkeypatch.setenv(litellm_model.LITELLM_CACHE_LOCATION_ENV, "/env/cache")

    assert litellm_model._get_litellm_cache_dir() == "/env/cache"
    assert litellm_model._get_litellm_cache_dir("/config/cache") == "/config/cache"


def test_set_litellm_cache_uses_configured_directory_and_can_reset(monkeypatch, tmp_path):
    created = []

    class FakeCache:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(litellm_model, "Cache", FakeCache, raising=False)
    monkeypatch.setattr(litellm_model, "LiteLLMCacheType", SimpleNamespace(DISK="disk"), raising=False)
    fake_litellm = SimpleNamespace()
    monkeypatch.setattr(litellm_model, "litellm", fake_litellm)

    litellm_model._set_litellm_cache(str(tmp_path))

    litellm_model._set_litellm_cache()

    assert created == [
        {"type": "disk", "disk_cache_dir": str(tmp_path)},
        {"type": "disk"},
    ]
    assert fake_litellm.cache is not None


def test_unconfigured_client_does_not_replace_process_cache(monkeypatch):
    # A second client using defaults must not reset another component's cache.
    monkeypatch.delenv(litellm_model.LITELLM_CACHE_LOCATION_ENV, raising=False)
    sentinel = object()
    fake_litellm = SimpleNamespace(cache=sentinel)
    monkeypatch.setattr(litellm_model, "litellm", fake_litellm)

    def fail_if_reconstructed(**kwargs):
        raise AssertionError("unconfigured client must reuse the global cache")

    monkeypatch.setattr(litellm_model, "Cache", fail_if_reconstructed, raising=False)
    litellm_model._configure_litellm_cache_if_requested(None)

    assert fake_litellm.cache is sentinel


def test_explicit_or_environment_cache_override_is_applied(monkeypatch):
    created = []

    class FakeCache:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(litellm_model, "Cache", FakeCache, raising=False)
    monkeypatch.setattr(litellm_model, "LiteLLMCacheType", SimpleNamespace(DISK="disk"), raising=False)
    fake_litellm = SimpleNamespace(cache=object())
    monkeypatch.setattr(litellm_model, "litellm", fake_litellm)
    monkeypatch.setenv(litellm_model.LITELLM_CACHE_LOCATION_ENV, "/env/cache")

    litellm_model._configure_litellm_cache_if_requested(None)
    litellm_model._configure_litellm_cache_if_requested("/config/cache")

    assert created == [
        {"type": "disk", "disk_cache_dir": "/env/cache"},
        {"type": "disk", "disk_cache_dir": "/config/cache"},
    ]
    assert isinstance(fake_litellm.cache, FakeCache)


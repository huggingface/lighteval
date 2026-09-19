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


def test_set_litellm_cache_uses_configured_directory(monkeypatch, tmp_path):
    created = {}

    class FakeCache:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(litellm_model, "Cache", FakeCache, raising=False)
    monkeypatch.setattr(litellm_model, "LiteLLMCacheType", SimpleNamespace(DISK="disk"), raising=False)
    fake_litellm = SimpleNamespace()
    monkeypatch.setattr(litellm_model, "litellm", fake_litellm)

    litellm_model._set_litellm_cache(str(tmp_path))

    assert created == {"type": "disk", "disk_cache_dir": str(tmp_path)}
    assert fake_litellm.cache is not None

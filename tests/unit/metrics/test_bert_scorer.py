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

from pathlib import Path

import pytest
import requests

from lighteval.metrics.imports.bert_scorer import BERTScorer


BASELINE_FILE_CONTENT = "LAYER,P,R,F\n0,0.1,0.2,0.3\n1,0.4,0.5,0.6\n"


class FakeResponse:
    def __init__(self, content: str):
        self.content = content.encode()

    def raise_for_status(self):
        return None


@pytest.fixture
def baseline_cache_dir(tmp_path, monkeypatch):
    """Redirects the baseline cache of the scorer to a temporary directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / ".cache/huggingface/lighteval/bertscore_baselines"


def test_missing_baseline_file_is_downloaded_and_cached(baseline_cache_dir, monkeypatch):
    requested_urls = []

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        return FakeResponse(BASELINE_FILE_CONTENT)

    monkeypatch.setattr(requests, "get", fake_get)

    scorer = BERTScorer(model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, num_layers=1)

    assert scorer.baseline_vals.tolist() == pytest.approx([0.4, 0.5, 0.6])
    assert requested_urls == [
        "https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/en/microsoft/deberta-large-mnli.tsv"
    ]
    cached_file = baseline_cache_dir / "en" / "microsoft" / "deberta-large-mnli.tsv"
    assert cached_file.read_text() == BASELINE_FILE_CONTENT

    # The cached file is reused on later runs, no second download
    reloaded_scorer = BERTScorer(
        model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, num_layers=1
    )
    assert reloaded_scorer.baseline_vals.tolist() == pytest.approx([0.4, 0.5, 0.6])
    assert len(requested_urls) == 1


def test_provided_baseline_file_is_used_as_is(baseline_cache_dir, monkeypatch):
    def fail_on_download(url, **kwargs):
        raise AssertionError(f"No baseline file should be downloaded, got a request to {url}")

    monkeypatch.setattr(requests, "get", fail_on_download)

    baseline_path = baseline_cache_dir / "custom_baseline.tsv"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(BASELINE_FILE_CONTENT)

    scorer = BERTScorer(
        model_type="microsoft/deberta-large-mnli",
        lang="en",
        rescale_with_baseline=True,
        num_layers=0,
        baseline_path=str(baseline_path),
    )

    assert scorer.baseline_vals.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_failed_baseline_download_raises_actionable_error(baseline_cache_dir, monkeypatch):
    def fake_get(url, **kwargs):
        raise requests.RequestException("connection error")

    monkeypatch.setattr(requests, "get", fake_get)

    scorer = BERTScorer(model_type="unknown-model", lang="en", rescale_with_baseline=True, num_layers=1)

    with pytest.raises(ValueError, match="Tiiiger/bert_score"):
        _ = scorer.baseline_vals

    # A failed download leaves no file behind
    assert list(Path(baseline_cache_dir).rglob("*")) == []

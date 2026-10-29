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

import pytest

from lighteval.metrics.imports.bert_scorer import BERTScorer


def test_missing_baseline_has_actionable_download_guidance(tmp_path, caplog):
    missing_baseline = tmp_path / "missing.tsv"
    expected_url = (
        "https://raw.githubusercontent.com/Tiiiger/bert_score/master/"
        "bert_score/rescale_baseline/en/microsoft/deberta-large-mnli.tsv"
    )
    scorer = BERTScorer(
        model_type="microsoft/deberta-large-mnli",
        lang="en",
        num_layers=9,
        rescale_with_baseline=True,
        baseline_path=str(missing_baseline),
        device="cpu",
    )

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as error:
        _ = scorer.baseline_vals

    assert str(missing_baseline) in caplog.text
    assert expected_url in caplog.text
    assert expected_url in str(error.value)
    assert "baseline_path" in str(error.value)

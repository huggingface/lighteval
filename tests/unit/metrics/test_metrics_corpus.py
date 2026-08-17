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

from lighteval.metrics.metrics_corpus import CorpusLevelTranslationMetric
from lighteval.metrics.sample_preparator import GenerativeCorpusMetricInput


# chrf is a similarity, ter an error rate, hence the different scores for a perfect first prediction
@pytest.mark.parametrize(
    ("metric_type", "metric_name", "expected_score"), [("chrf", "CHRF", 100.0), ("ter", "TER", 0.0)]
)
def test_translation_metric_with_several_predictions(metric_type, metric_name, expected_score, caplog):
    """Sampling metrics give several predictions per sample, only the first one is scored."""
    items = [GenerativeCorpusMetricInput(golds=["the cat sat on the mat"], preds=["the cat sat on the mat", "a dog"])]

    with caplog.at_level(logging.INFO, logger="lighteval.metrics.metrics_corpus"):
        score = CorpusLevelTranslationMetric(metric_type).compute_corpus(items)

    assert score == pytest.approx(expected_score)
    assert f"sacrebleu.{metric_name}" in caplog.text

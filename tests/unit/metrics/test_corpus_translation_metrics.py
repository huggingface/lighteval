# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from itertools import zip_longest

import pytest

from lighteval.metrics.metrics_corpus import CorpusLevelTranslationMetric
from lighteval.metrics.sample_preparator import GenerativeCorpusMetricInput
from lighteval.utils.utils import as_list


def _transpose_references(items: list[GenerativeCorpusMetricInput]) -> list[list[str | None]]:
    per_sample_references = [as_list(item.golds) for item in items]
    return [list(ref_group) for ref_group in zip_longest(*per_sample_references, fillvalue=None)]


def _first_prediction_per_sample(items: list[GenerativeCorpusMetricInput]) -> list[str]:
    return [as_list(item.preds)[0] for item in items]


@pytest.mark.parametrize("metric_type", ["chrf", "chrf++", "ter"])
def test_translation_metrics_use_all_hypotheses(metric_type: str):
    items = [
        GenerativeCorpusMetricInput(golds=["GOOD"], preds=["GOOD"]),
        GenerativeCorpusMetricInput(golds=["REF2"], preds=["PRED2"]),
    ]
    metric = CorpusLevelTranslationMetric(metric_type=metric_type)

    hypotheses = _first_prediction_per_sample(items)
    expected_references = _transpose_references(items)
    wrong_orientation_references = [item.golds for item in items]

    expected_score = metric.get_metric().corpus_score(hypotheses=hypotheses, references=expected_references).score
    wrong_score = (
        metric.get_metric().corpus_score(hypotheses=hypotheses, references=wrong_orientation_references).score
    )
    actual_score = metric.compute_corpus(items)

    assert actual_score == pytest.approx(expected_score)
    assert wrong_score != pytest.approx(expected_score)


@pytest.mark.parametrize("metric_type", ["chrf", "chrf++", "ter"])
def test_translation_metrics_support_variable_reference_counts(metric_type: str):
    items = [
        GenerativeCorpusMetricInput(golds=["the cat sits", "cat is sitting"], preds=["the cat sits"]),
        GenerativeCorpusMetricInput(golds=["goodbye"], preds=["hello"]),
    ]
    metric = CorpusLevelTranslationMetric(metric_type=metric_type)

    hypotheses = _first_prediction_per_sample(items)
    references = _transpose_references(items)
    expected_score = metric.get_metric().corpus_score(hypotheses=hypotheses, references=references).score

    assert metric.compute_corpus(items) == pytest.approx(expected_score)

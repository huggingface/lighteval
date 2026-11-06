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

import pytest
import sklearn.metrics

from lighteval.metrics.metrics_corpus import CorpusLevelF1Score
from lighteval.metrics.sample_preparator import LogprobCorpusMetricInput


def _f1(golds: list[int], preds: list[int], average: str, num_classes: int = 3) -> float:
    items = [LogprobCorpusMetricInput(golds=gold, preds=pred) for gold, pred in zip(golds, preds)]
    return CorpusLevelF1Score(average=average, num_classes=num_classes).compute_corpus(items)


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_multiclass_mixed_corpus_matches_sklearn(average: str):
    """Per-class binary f1 averaged together is not the requested multiclass average."""
    golds = [0, 1, 2, 0, 1, 2]
    preds = [0, 0, 0, 0, 0, 0]
    expected = sklearn.metrics.f1_score(golds, preds, average=average, labels=[0, 1, 2])
    assert _f1(golds, preds, average) == pytest.approx(expected)


def test_micro_and_macro_differ():
    """The mixed corpus is the case where micro != macro, so both must be checked separately."""
    golds = [0, 1, 2, 0, 1, 2]
    preds = [0, 0, 0, 0, 0, 0]
    micro = _f1(golds, preds, "micro")
    macro = _f1(golds, preds, "macro")
    assert micro != pytest.approx(macro)
    assert micro == pytest.approx(2 / 6)
    assert macro == pytest.approx(1 / 6)


def test_all_predictions_zero():
    """All zero predictions: micro f1 is 1.0, classes 1 and 2 are absent and score 0 in macro."""
    golds = [0, 0, 0, 0]
    preds = [0, 0, 0, 0]
    assert _f1(golds, preds, "micro") == pytest.approx(1.0)
    assert _f1(golds, preds, "macro") == pytest.approx(1 / 3)


def test_single_class_prediction():
    """Every sample predicted as class 1: micro = 2TP/(2TP+FP+FN) over all classes."""
    golds = [0, 1, 2]
    preds = [1, 1, 1]
    assert _f1(golds, preds, "micro") == pytest.approx(2 * 1 / (2 * 1 + 2 + 2))
    assert _f1(golds, preds, "macro") == pytest.approx(1 / 6)


def test_class_absent_from_corpus():
    """Class 2 never occurs: it must not be averaged in as an extra binary problem."""
    golds = [0, 1, 0, 1]
    preds = [0, 1, 1, 0]
    assert _f1(golds, preds, "micro") == pytest.approx(0.5)
    assert _f1(golds, preds, "macro") == pytest.approx(1 / 3)

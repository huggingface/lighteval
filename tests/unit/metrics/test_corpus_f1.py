# MIT License

# Copyright (c) 2026 The HuggingFace Team

from copy import deepcopy

import pytest

from lighteval.metrics.metrics_corpus import CorpusLevelF1Score
from lighteval.metrics.sample_preparator import LogprobCorpusMetricInput
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.glue import cb_prompt, super_glue_cb


def cb_scores(golds, predictions):
    metric = deepcopy(next(metric for metric in super_glue_cb.metrics if metric.metric_name == "mf1"))
    items = []
    for gold, prediction in zip(golds, predictions):
        doc = cb_prompt({"premise": "A fixed premise.", "hypothesis": "A fixed hypothesis", "label": gold})
        response = ModelResponse(logprobs=[-0.1 if label == prediction else -3.0 for label in range(3)])
        items.append(metric.compute_sample(doc=doc, model_response=response)["mf1"])
    assert [item.golds for item in items] == golds
    assert [item.preds for item in items] == predictions
    assert len(items) == len(golds)
    return metric.get_corpus_aggregations()["mf1"](items)


@pytest.mark.parametrize(
    "golds,predictions,expected",
    [
        ([0, 1, 2], [1, 2, 0], 0.0),
        ([0, 1, 2], [0, 1, 2], 1.0),
        # Per-class F1: (2/3, 0, 0), averaged over the three observed labels.
        ([0, 0, 1, 2], [0, 0, 0, 0], 2 / 9),
        # Per-class F1: (4/5, 2/3, 1).
        ([0, 0, 0, 1, 2], [0, 0, 1, 1, 2], 37 / 45),
        ([0, 1, 0, 1], [1, 0, 1, 0], 0.0),
        # Absent labels must not dilute an otherwise perfect limited run.
        ([0, 0, 0], [0, 0, 0], 1.0),
    ],
)
def test_cb_macro_f1(golds, predictions, expected):
    assert cb_scores(golds, predictions) == pytest.approx(expected)


def test_cb_macro_f1_preserves_order_and_repeated_samples():
    golds, predictions = [0, 0, 1, 2], [0, 0, 0, 0]
    expected = 2 / 9
    assert cb_scores(golds, predictions) == pytest.approx(expected)
    assert cb_scores(golds[::-1], predictions[::-1]) == pytest.approx(expected)
    assert cb_scores(golds * 2, predictions * 2) == pytest.approx(expected)


@pytest.mark.parametrize("average,expected", [("macro", 2 / 9), ("micro", 1 / 2), ("weighted", 1 / 3)])
def test_multiclass_f1_averages(average, expected):
    # Only label 0 has nonzero F1 (2/3), with support 2 out of 4 samples.
    items = [LogprobCorpusMetricInput(golds=g, preds=p) for g, p in zip([0, 0, 1, 2], [0, 0, 0, 0])]
    assert CorpusLevelF1Score(average=average, num_classes=3).compute_corpus(items) == pytest.approx(expected)


@pytest.mark.parametrize("average,expected", [("macro", 11 / 15), ("micro", 3 / 4), ("weighted", 11 / 15)])
def test_binary_f1_averages_unchanged(average, expected):
    items = [LogprobCorpusMetricInput(golds=g, preds=p) for g, p in zip([0, 0, 1, 1], [0, 1, 1, 1])]
    assert CorpusLevelF1Score(average=average).compute_corpus(items) == pytest.approx(expected)


@pytest.mark.parametrize(
    "num_classes,golds,predictions,expected",
    [
        # Preserve the existing binary max-per-class behavior; its positive-class
        # interpretation is a separate change from CB's multiclass macro-F1.
        (2, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 5 / 7),
        # Preserve the legacy None reduction over both binary labels per class:
        # mean(0, 2/3, 6/7, 0, 6/7, 0) = 25/63.
        (3, [0, 0, 1, 2], [0, 0, 0, 0], 25 / 63),
    ],
)
def test_f1_none_compatibility(num_classes, golds, predictions, expected):
    items = [LogprobCorpusMetricInput(golds=g, preds=p) for g, p in zip(golds, predictions)]
    assert CorpusLevelF1Score(average=None, num_classes=num_classes).compute_corpus(items) == pytest.approx(expected)

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
# DEALINGS IN THE SOFTWARE.

import numpy as np
import pytest

from lighteval.metrics.bayes_at_n import bayes_at_n
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_corpus import BayesAtNCorpus
from lighteval.metrics.metrics_sample import BayesAtN
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def test_bayes_at_n_multicategory_with_prior():
    R = np.array([[0, 1, 2, 2, 1], [1, 1, 0, 2, 2]])
    weights = np.array([0.0, 0.5, 1.0])
    prior = np.array([[0, 2], [1, 2]])

    mu, sigma = bayes_at_n(R, weights=weights, prior=prior)

    assert mu == pytest.approx(0.575)
    assert sigma == pytest.approx(0.084275, abs=1e-6)


def test_bayes_at_n_multicategory_without_prior():
    R = np.array([[0, 1, 2, 2, 1], [1, 1, 0, 2, 2]])
    weights = np.array([0.0, 0.5, 1.0])

    mu, sigma = bayes_at_n(R, weights=weights)

    assert mu == pytest.approx(0.5625)
    assert sigma == pytest.approx(0.091998, abs=1e-6)


def test_bayes_at_n_binary_defaults_weights():
    mu, sigma = bayes_at_n([[0, 1, 1], [1, 1, 0]])

    assert mu == pytest.approx(0.6)
    assert sigma == pytest.approx(0.1414213562373095)


def test_bayes_at_n_requires_weights_for_multicategory():
    with pytest.raises(ValueError, match="pass weights"):
        bayes_at_n([[0, 1, 2]])


def test_bayes_at_n_validates_categories_and_prior_shape():
    with pytest.raises(ValueError, match="R entries"):
        bayes_at_n([[0, 2]], weights=[0.0, 1.0])

    with pytest.raises(ValueError, match="integer category ids"):
        bayes_at_n([[0.5, 1.0]])

    with pytest.raises(ValueError, match="same number of rows"):
        bayes_at_n([[0, 1], [1, 0]], prior=[[0, 1, 0]])


def test_bayes_at_n_corpus_aggregator_uses_all_rows():
    items = [
        {"scores": [0, 1, 2, 2, 1], "weights": [0.0, 0.5, 1.0], "prior": [[0, 2], [1, 2]]},
        {"scores": [1, 1, 0, 2, 2], "weights": [0.0, 0.5, 1.0], "prior": [[0, 2], [1, 2]]},
    ]

    assert BayesAtNCorpus("mu").compute_corpus(items) == pytest.approx(0.575)
    assert BayesAtNCorpus("sigma").compute_corpus(items) == pytest.approx(0.084275, abs=1e-6)


def test_bayes_at_n_sample_metric_and_registration():
    metric = Metrics.bayes_at_n(sample_params={"n": 5})
    metric_name, sigma_name = metric.metric_name
    docs = [
        Doc(query="q1", choices=["A"], gold_index=0),
        Doc(query="q2", choices=["A"], gold_index=0),
    ]
    responses = [
        ModelResponse(text=["A", "B", "A", "A", "B"]),
        ModelResponse(text=["B", "A", "B", "B", "A"]),
    ]

    sample_outputs = [metric.compute_sample(doc=doc, model_response=response) for doc, response in zip(docs, responses)]
    expected_rows = [[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]]
    expected_mu, expected_sigma = bayes_at_n(expected_rows)

    assert sample_outputs[0][metric_name]["scores"] == expected_rows[0]
    aggregations = metric.get_corpus_aggregations()
    assert aggregations[metric_name]([output[metric_name] for output in sample_outputs]) == pytest.approx(expected_mu)
    assert aggregations[sigma_name]([output[sigma_name] for output in sample_outputs]) == pytest.approx(expected_sigma)


def test_bayes_at_n_rejects_continuous_sample_scores():
    metric = BayesAtN(n=1, sample_scoring_function=lambda doc, response: 0.5)
    doc = Doc(query="q", choices=["A"], gold_index=0)
    response = ModelResponse(text=["A"])

    with pytest.raises(ValueError, match="integer category ids"):
        metric.compute(doc=doc, model_response=response)

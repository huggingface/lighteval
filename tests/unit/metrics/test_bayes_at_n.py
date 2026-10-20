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

import json
import math
from fractions import Fraction
from types import ModuleType

import numpy as np
import pytest

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.info_loggers import MetricsLogger
from lighteval.metrics import apply_metric
from lighteval.metrics.bayes_at_n import bayes_at_n
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_corpus import BayesAtNCorpus
from lighteval.metrics.metrics_sample import BayesAtN
from lighteval.models.dummy.dummy_model import DummyModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import Pipeline
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod


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
        {"scores": [0, 1, 2, 2, 1], "weights": [0.0, 0.5, 1.0], "prior": [0, 2]},
        {"scores": [1, 1, 0, 2, 2], "weights": [0.0, 0.5, 1.0], "prior": [1, 2]},
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

    sample_outputs = [
        metric.compute_sample(doc=doc, model_response=response) for doc, response in zip(docs, responses)
    ]
    expected_rows = [[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]]
    expected_mu, expected_sigma = 0.5, math.sqrt(3 / 196)

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


@pytest.mark.parametrize(
    "outcomes,expected_mean,expected_variance",
    [
        ([[0, 0], [1, 1]], Fraction(1, 2), Fraction(3, 160)),
        ([[0, 1], [0, 1]], Fraction(1, 2), Fraction(1, 40)),
        ([[0]], Fraction(1, 3), Fraction(1, 18)),
        ([[1]], Fraction(2, 3), Fraction(1, 18)),
    ],
)
def test_independent_exact_binary_cases(outcomes, expected_mean, expected_variance):
    mean, sigma = bayes_at_n(outcomes)
    assert mean == pytest.approx(float(expected_mean))
    assert sigma**2 == pytest.approx(float(expected_variance))


def test_representative_categorical_workload_against_rational_moments():
    rng = np.random.default_rng(92)
    outcomes = rng.integers(0, 4, size=(80, 16))
    prior = rng.integers(0, 4, size=(80, 3))
    weights = [Fraction(-2), Fraction(1, 4), Fraction(1, 4), Fraction(3)]
    means, variances = [], []
    for row, prior_row in zip(outcomes.tolist(), prior.tolist()):
        counts = [1 + row.count(k) + prior_row.count(k) for k in range(4)]
        total = sum(counts)
        mean = sum(Fraction(count, total) * weight for count, weight in zip(counts, weights))
        variance = (sum(Fraction(count, total) * weight**2 for count, weight in zip(counts, weights)) - mean**2) / (
            total + 1
        )
        means.append(mean)
        variances.append(variance)
    result = bayes_at_n(outcomes, [float(weight) for weight in weights], prior)
    assert result[0] == pytest.approx(float(sum(means) / 80), rel=1e-14)
    assert result[1] ** 2 == pytest.approx(float(sum(variances) / 80**2), rel=1e-14)
    assert bayes_at_n(outcomes[::-1, ::-1], [float(weight) for weight in weights], prior[::-1]) == pytest.approx(
        result
    )


@pytest.mark.parametrize(
    "bad", [[[0j, 1]], [["0", "1"]], [[np.nan, 1]], [[np.inf]], [[0.2]], [[2**63]], [[-1]], [[0], [0, 1]], []]
)
def test_invalid_outcomes(bad):
    with pytest.raises(ValueError):
        bayes_at_n(bad)


@pytest.mark.parametrize("weights", [[0j, 1], ["0", "1"], [0, np.inf], [], [[0, 1]]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        bayes_at_n([[0, 1]], weights=weights)


def test_prior_shape_and_stable_rewards():
    outcomes = [[0, 1], [1, 1]]
    assert bayes_at_n(outcomes, prior=[1, 0]) == pytest.approx(bayes_at_n(outcomes, prior=[[1], [0]]))
    with pytest.raises(ValueError, match="same number of rows"):
        bayes_at_n(outcomes, prior=[[1, 0]])
    assert bayes_at_n(outcomes, weights=[1e155, 1e155]) == (1e155, 0)
    mean, sigma = bayes_at_n(outcomes, weights=[0, 1e155])
    expected_mean, expected_sigma = bayes_at_n(outcomes)
    assert mean / 1e155 == pytest.approx(expected_mean)
    assert sigma / 1e155 == pytest.approx(expected_sigma)
    assert bayes_at_n([[0, 1]], weights=[-1e308, 1e308]) == pytest.approx((0, 1e308 / math.sqrt(5)))
    assert bayes_at_n([[0, 1]], weights=[1e12, 1e12 + 1])[1] == pytest.approx(math.sqrt(1 / 20))


@pytest.mark.parametrize("n", [True, 0, -1, 1.5])
def test_invalid_sample_count(n):
    with pytest.raises(ValueError, match="positive integer"):
        BayesAtN(n=n).num_samples()


def test_missing_predictions_and_fractional_scores():
    doc = Doc(query="q", choices=["A"], gold_index=0)
    with pytest.raises(ValueError, match="at least n"):
        BayesAtN(n=2).compute(doc, ModelResponse(text=["A"]))
    with pytest.raises(ValueError):
        BayesAtN().compute(doc, ModelResponse())


def test_prior_rows_follow_docs_through_serialization_and_reordering():
    docs = [
        Doc(query="q1", choices=[" A "], gold_index=0, specific={"prior": [0, 0]}),
        Doc(query="q2", choices=[" A "], gold_index=0, specific={"prior": [1, 1]}),
    ]
    responses = [ModelResponse(text=["A", "B"]), ModelResponse(text=["B", "B"])]
    metric = Metrics.bayes_at_n(sample_params={"n": 2, "prior": lambda doc: doc.specific["prior"]})
    outputs = apply_metric(responses, docs, [metric])
    payloads = json.loads(json.dumps([output["bayes@n"] for output in outputs]))
    assert docs[0].choices == [" A "]
    assert payloads[0]["prior"] == [0, 0]
    assert payloads[1]["prior"] == [1, 1]
    # Beta(2,4) and Beta(3,3), each question keeps its own prior row.
    assert BayesAtNCorpus("mu").compute_corpus(payloads[::-1]) == pytest.approx(5 / 12)
    assert BayesAtNCorpus("sigma").compute_corpus(payloads[::-1]) ** 2 == pytest.approx(Fraction(17, 1008))


@pytest.mark.parametrize(
    "items",
    [
        [{"scores": [0]}, {"scores": [0, 1]}],
        [{"scores": [0], "weights": [0, 1]}, {"scores": [1]}],
        [{"scores": [0], "prior": [0]}, {"scores": [1]}],
        [{"scores": [0], "prior": [[0], [1]]}],
        [{"scores": [0], "prior": [0]}, {"scores": [1], "prior": []}],
    ],
)
def test_incompatible_corpus_payloads(items):
    with pytest.raises(ValueError):
        BayesAtNCorpus("mu").compute_corpus(items)


def _task(name, metric):
    return LightevalTask(
        LightevalTaskConfig(
            name=name,
            prompt_function=lambda row, task_name: None,
            hf_repo="unused",
            hf_subset="unused",
            metrics=[metric],
        )
    )


def test_public_task_path_and_independent_task_sd_summaries():
    logger = MetricsLogger()
    tasks = {}
    for suffix, rows in (("a", [[0, 0], [1, 1]]), ("b", [[0, 1], [0, 1], [1, 1]])):
        metric = Metrics.bayes_at_n(sample_params={"n": 2})
        task = _task(f"bayes_fixture:{suffix}", metric)
        assert max(task.num_samples) == 2
        tasks[task.full_name] = task
        docs = [Doc(query=str(i), choices=["A"], gold_index=0) for i in range(len(rows))]
        responses = [ModelResponse(text=["A" if grade else "B" for grade in row]) for row in rows]
        for output in apply_metric(responses, docs, task.metrics):
            logger.log(task.full_name, json.loads(json.dumps(output)))
    logger.aggregate(tasks)
    assert logger.metric_aggregated["bayes_fixture:a|0"]["bayes@n_sigma"] ** 2 == pytest.approx(3 / 160)
    # Task b has posterior row variances 1/20, 1/20, 3/80.
    expected_sigma = math.sqrt((3 / 160 + 11 / 720) / 4)
    for name in ("bayes_fixture:_average|0", "all"):
        assert logger.metric_aggregated[name]["bayes@n"] == pytest.approx(13 / 24)
        assert logger.metric_aggregated[name]["bayes@n_sigma"] == pytest.approx(expected_sigma)
        assert "bayes@n_sigma_stderr" not in logger.metric_aggregated[name]


def test_registry_cli_configures_grouped_sampling_metric(monkeypatch):
    custom = ModuleType("bayes_fixture")
    custom.TASKS_TABLE = [_task("bayes_fixture", Metrics.bayes_at_n.value).config]
    monkeypatch.setattr(Registry, "_load_from_files", staticmethod(lambda *args: {}))
    monkeypatch.setattr(Registry, "_load_from_subdirs", staticmethod(lambda *args: {}))
    registry = Registry(tasks="bayes_fixture@n=3|0", custom_tasks=custom)
    task = registry.load_tasks()["bayes_fixture|0"]
    assert task.metrics[0].sample_level_fn.n == 3
    assert max(task.num_samples) == 3


def test_math_registered_scorer_uses_normalized_candidate_answers():
    metric = Metrics.bayes_at_n_math(sample_params={"n": 2})
    doc = Doc(query="Compute 1+1.", choices=["2"], gold_index=0)
    output = metric.compute_sample(doc=doc, model_response=ModelResponse(text=[r"\boxed{2}", r"\boxed{3}"]))
    assert output["math-bayes@n"]["scores"] == [1, 0]


def test_pipeline_grouped_options_and_export(tmp_path):
    metric = Metrics.bayes_at_n(sample_params={"n": 2})
    task = _task("bayes_fixture", metric)
    tracker = EvaluationTracker(output_dir=str(tmp_path), save_details=True)
    tracker.general_config_logger.log_model_info(DummyModelConfig(model_name="bayes_fixture"))
    tracker.task_config_logger.log({task.full_name: task})
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._metric_options = {"bayes@n": {"num_samples": 2}}
    pipeline._update_num_samples([task])
    assert task.num_samples == [2]
    pipeline.tasks_dict = {task.full_name: task}
    pipeline.sampling_docs = {
        SamplingMethod.GENERATIVE: [Doc(task_name=task.full_name, query="q", choices=["A"], gold_index=0)]
    }
    pipeline.evaluation_tracker = tracker
    pipeline._compute_metrics({SamplingMethod.GENERATIVE: [ModelResponse(text=["A", "B"])]})
    tracker.metrics_logger.aggregate(pipeline.tasks_dict)
    tracker.details_logger.aggregate()
    tracker.save()
    saved = json.loads(next(tmp_path.rglob("results_*.json")).read_text())
    assert saved["results"][task.full_name]["bayes@n"] == pytest.approx(0.5)
    assert saved["results"][task.full_name]["bayes@n_sigma"] == pytest.approx(math.sqrt(1 / 20))
    import pyarrow.parquet as pq

    details = pq.read_table(next(tmp_path.rglob("*.parquet"))).to_pylist()
    assert details[0]["metric"]["bayes@n"]["scores"] == [1, 0]


def test_custom_scorer_receives_doc_and_candidate_metadata():
    seen = []

    def score(doc, response):
        seen.append(response.final_text[0])
        return int(response.logprobs[0] > doc.specific["cutoff"])

    metric = BayesAtN(n=2, strip_strings=True, sample_scoring_function=score)
    doc = Doc(query="q", choices=[" A "], gold_index=0, specific={"cutoff": -1})
    response = ModelResponse(text=["raw1", "raw2"], text_post_processed=[" A ", " B "], logprobs=[-0.1, -2])
    assert metric.compute(doc, response)["bayes@n"]["scores"] == [1, 0]
    assert seen == ["A", "B"]
    assert response.final_text == [" A ", " B "]
    assert doc.choices == [" A "]

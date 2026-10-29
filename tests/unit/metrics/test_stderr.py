# MIT License

# Copyright (c) 2026 ningyixi

import math
import multiprocessing as mp
import statistics

import pytest

from lighteval.metrics.utils.stderr import get_stderr_function


class _RecordingMetric:
    __name__ = "recording_metric"

    def __init__(self, draws):
        self.draws = draws

    def __call__(self, sample):
        value = statistics.mean(sample)
        self.draws.append(value)
        return value


@pytest.mark.parametrize("number_experiments", [25, 1000, 1001, 1500])
def test_bootstrap_uses_every_requested_experiment(monkeypatch, number_experiments):
    monkeypatch.setattr(mp, "cpu_count", lambda: 2)
    with mp.Manager() as manager:
        draws = manager.list()
        stderr = get_stderr_function(_RecordingMetric(draws), number_experiments=number_experiments)

        result = stderr([0, 1, 2, 4])

        assert len(draws) == number_experiments
        assert result == pytest.approx(statistics.stdev(draws))


def _accuracy(sample):
    return sum(sample) / len(sample)


@pytest.mark.parametrize("number_experiments", [1000, 4000])
def test_bootstrap_standard_error_matches_bernoulli_sampling_variance(monkeypatch, number_experiments):
    monkeypatch.setattr(mp, "cpu_count", lambda: 2)
    population = [0, 1] * 250
    stderr = get_stderr_function(_accuracy, number_experiments=number_experiments)

    # Resampling this empirical Bernoulli population gives variance p(1-p)/n
    # for its accuracy, independent of the number of bootstrap replicates.
    assert stderr(population) == pytest.approx(math.sqrt(0.5 * 0.5 / len(population)), rel=0.08)


def test_mean_aggregation_keeps_analytic_standard_error():
    population = [0, 1, 2, 4]
    stderr = get_stderr_function(statistics.mean)

    assert stderr(population) == pytest.approx(statistics.stdev(population) / math.sqrt(len(population)))

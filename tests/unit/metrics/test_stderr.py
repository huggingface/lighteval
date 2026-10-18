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
        assert result == pytest.approx(statistics.stdev(draws) / math.sqrt(number_experiments))

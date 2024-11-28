# MIT License

# Copyright (c) 2024 Eleuther AI and The HuggingFace Team

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

# This section of the code comes almost entirely from the harness
# We kept it because it's very fast - however, we renamed the variables
# and added documentation

import math
import random
from typing import Callable, Optional

import numpy as np
from scipy.stats import bootstrap
from tqdm import tqdm

from lighteval.logging.hierarchical_logger import hlog


def _stddev(arr):
    mu = np.mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return _stddev(arr) / math.sqrt(len(arr))


class _bootstrap_internal:
    def __init__(self, number_draws: int, metric: Optional[Callable] = None):
        self.number_draws = number_draws
        self.metric = metric

    def __call__(self, cur_experiment):
        # Creates number_draws samplings (with replacement) of the population by iterating on a given seed
        population, seed = cur_experiment
        rnd = random.Random()
        rnd.seed(seed)
        samplings = []
        for _ in range(self.number_draws):
            if self.metric is None:
                # For sample-level metrics, just compute mean of sampled precomputed values
                sampled_values = rnd.choices(population, k=len(population))
                samplings.append(np.mean(sampled_values))
            else:
                # For corpus-level metrics, recompute metric on sampled values
                sampled_values = rnd.choices(population, k=len(population))
                samplings.append(self.metric(sampled_values))
        return samplings


def bootstrap_stderr(
    population: list,
    number_experiments: int,
    metric: Optional[Callable] = None,
) -> float:
    """Bootstraps the stderr by resampling.

    For sample-level metrics, resamples from precomputed values to avoid recomputing heavy metrics.
    For corpus-level metrics, recomputes metric on resampled values.

    Args:
        population (list): List of values to bootstrap from
        number_experiments (int): Total number of bootstrap iterations
        metric (Optional[Callable]): Metric function for corpus-level metrics

    Returns:
        float: Standard error estimate from bootstrap
    """
    if metric is None:
        # For sample-level metrics, verify values are numeric
        try:
            population = [float(x) for x in population]
        except (TypeError, ValueError) as e:
            raise TypeError(f"All values must be numeric for sample-level bootstrap. Got error: {e}")

    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())

    res = []
    number_draws = min(1000, number_experiments)
    number_seeds = number_experiments // number_draws

    for cur_bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(number_draws=number_draws, metric=metric),
            ((population, seed) for seed in range(number_seeds)),
        ),
        total=number_seeds,
    ):
        res.extend(cur_bootstrap)

    pool.close()
    return mean_stderr(res)


def get_stderr_function(
    metric_values: list,
    number_experiments: int = 1000,
    aggregation: Optional[Callable] = None,
) -> Optional[Callable]:
    """Get the appropriate stderr function for the given metric values.

    Args:
        metric_values (list): List of values to compute stderr from
        number_experiments (int): Number of bootstrap iterations
        aggregation (Optional[Callable]): Aggregation function for corpus-level metrics

    Returns:
        Optional[Callable]: Function to compute standard error, or None if not possible
    """
    if len(metric_values) <= 1:
        return None

    # For sample-level metrics
    if isinstance(metric_values[0], (int, float)) or aggregation is None:
        try:
            _ = [float(x) for x in metric_values]
            return lambda _: bootstrap_stderr(population=metric_values, number_experiments=number_experiments)
        except (TypeError, ValueError):
            return None

    # For corpus-level metrics
    try:
        return lambda _: bootstrap_stderr(
            population=metric_values, number_experiments=number_experiments, metric=aggregation
        )
    except Exception:
        return None


def bootstrap_stderr_scipy(metric: Callable, population: list, number_experiments: int = 1000):
    """Simulates resampling (draw with replacement, of the same size as the orig set) n times from the results population
    to compute the distance between the simulated resampling distribution and the actual distribution.
    Same as bootstrap_stderr, but uses scipy.
    It's kept for archive, as it overflows for big datasets
    """
    hlog(f"Bootstrapping {metric.__name__}'s stderr.")
    res = bootstrap(
        data=[population],
        statistic=metric,
        n_resamples=number_experiments,
        confidence_level=0.95,
        method="BCa",
    )
    return res.standard_error

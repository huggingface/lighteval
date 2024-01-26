# This section of the code comes almost entirely from the harness
# We kept it because it's very fast - however, we renamed the variables
# and added documentation

import math
import random
from typing import Callable

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
    def __init__(self, metric: Callable, number_draws: int):
        self.metric = metric
        self.number_draws = number_draws

    def __call__(self, cur_experiment):
        # Creates number_draws samplings (with replacement) of the population by iterating on a given seed
        population, seed = cur_experiment
        rnd = random.Random()
        rnd.seed(seed)
        samplings = []
        for _ in range(self.number_draws):
            samplings.append(self.metric(rnd.choices(population, k=len(population))))
        return samplings


def bootstrap_stderr(metric: Callable, population: list, number_experiments: int):
    """Bootstraps the stderr of the given metric for the given population of samples,
    by sampling said population for number_experiments and recomputing the metric on the
    different samplings.
    """
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())

    res = []
    number_draws = min(1000, number_experiments)
    # We change the seed every 1000 re-samplings
    # and do the experiment 1000 re-samplings at a time
    number_seeds = number_experiments // number_draws

    hlog(f"Bootstrapping {metric.__name__}'s stderr.")
    for cur_bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(metric=metric, number_draws=number_draws),
            ((population, seed) for seed in range(number_seeds)),
        ),
        total=number_seeds,
    ):
        # sample w replacement
        res.extend(cur_bootstrap)

    pool.close()
    return mean_stderr(res)


def get_stderr_function(aggregation: Callable, number_experiments: int = 1000):
    # Mean stderr can be computed trivially
    if "mean" in aggregation.__name__:
        return mean_stderr

    # For other metrics, we bootstrap the stderr by sampling
    try:
        return lambda population: bootstrap_stderr(
            metric=aggregation, population=population, number_experiments=number_experiments
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

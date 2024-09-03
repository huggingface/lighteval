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

from typing import Callable

import numpy as np

from lighteval.metrics.metrics_sample import LoglikelihoodAcc, NormalizedMultiChoiceProbability, Probability
from lighteval.metrics.normalizations import LogProbNormalization, LogProbPMINorm, LogProbTokenNorm
from lighteval.metrics.utils import MetricCategory, MetricUseCase, SampleLevelMetric


def loglikelihood_acc_metric(normalization: LogProbNormalization | None = None) -> SampleLevelMetric:
    """
    Creates a accuracy (loglikelihood) metric, which returns accuracy given normalization.
    """

    normalization_str = normalization.name if normalization else ""
    metric_name = f"acc_{normalization_str}"
    return SampleLevelMetric(
        metric_name=metric_name,
        sample_level_fn=LoglikelihoodAcc(logprob_normalization=normalization).compute,
        category=MetricCategory.MULTICHOICE
        if not normalization == LogProbPMINorm()
        else MetricCategory.MULTICHOICE_PMI,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def normalized_multi_choice_prob_metric(
    normalization: LogProbNormalization | None = None,
    aggregation_function: Callable[[np.ndarray], float] = np.max,
) -> SampleLevelMetric:
    """
    Creates a normalized multi-choice probability metric, which returns the probability of the gold choice / sum of probabilities of all choices (after logprobs are normalized).
    """

    normalization_str = normalization.name if normalization else ""
    metric_name = "_".join(filter(None, ["normalized_mc_prob_", normalization_str]))

    return SampleLevelMetric(
        metric_name=metric_name,
        sample_level_fn=NormalizedMultiChoiceProbability(
            log_prob_normalization=normalization, aggregation_function=aggregation_function
        ).compute,
        category=MetricCategory.MULTICHOICE
        if not normalization == LogProbPMINorm()
        else MetricCategory.MULTICHOICE_PMI,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def probability_metric(
    normalization: LogProbTokenNorm | None = None,
    aggregation_function: Callable[[np.ndarray], float] = np.max,
) -> SampleLevelMetric:
    """
    Creates a probability metric, which returns the probability of the gold choice given normalization.
    """

    normalization_str = normalization.name if normalization else ""
    metric_name = "_".join(filter(None, ["prob", normalization_str]))

    return SampleLevelMetric(
        metric_name=metric_name,
        sample_level_fn=Probability(normalization=normalization, aggregation_function=aggregation_function).compute,
        category=MetricCategory.TARGET_PERPLEXITY,
        use_case=MetricUseCase.PERPLEXITY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

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

"""This module manages all the metrics occurring at the corpus level.
Some metrics (such as corpus BLEU) are not computed at the individual item level, but over all the corpus.
A number of these aggregations come from the EleutherAIHarness
"""
import math

import numpy as np
import sacrebleu
import sklearn.metrics

from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from lighteval.utils.utils import as_list


# General aggregations
def matthews_corrcoef(items: list[GenerativeCorpusMetricInput]) -> float:
    """Computes the Matthews Correlation Coefficient, using scikit learn ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)).

    Args:
        items (list[dict]): List of GenerativeCorpusMetricInput

    Returns:
        float: Score
    """
    golds = [i.golds for i in items]
    preds = [i.preds for i in items]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


class CorpusLevelF1Score:
    def __init__(self, average: str, num_classes: int = 2):
        """Stores the relevant parameters for the task's corpus level f1 score.

        Args:
            average (str): Method to use to compute the f1 score. Can be weighted, macro, micro.
            num_classes (int, optional): Num of possible choice classes. Defaults to 2. If this parameter is above 2, we'll compute multi f1 corpus score
        """
        if average not in ["weighted", "macro", "micro", None]:
            raise ValueError(
                f"A CorpusLevelF1Score must be initialized with weighted, macro, micro, or None as an average function. {average} was used."
            )
        self.average = average
        self.num_classes = num_classes

    def compute(self, items: list[LogprobCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items, by using the scikit learn implementation."""
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        # Single f1
        if self.num_classes == 2:
            fscore = sklearn.metrics.f1_score(golds, preds, average=self.average)
            return np.max(fscore)

        # Multi f1
        f1s = []
        for i in range(self.num_classes):
            f1s.append(sklearn.metrics.f1_score(y_true=golds == i, y_pred=preds == i))
        return float(np.mean(f1s))


class CorpusLevelTranslationMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        if metric_type == "bleu":
            self.metric = sacrebleu.corpus_bleu
        elif metric_type == "chrf":
            self.metric = sacrebleu.corpus_chrf
        elif metric_type == "ter":
            self.metric = sacrebleu.corpus_ter
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the metric score over all the corpus generated items, by using the sacrebleu implementation."""
        golds = [i.golds for i in items]
        preds = [as_list(i.preds) for i in items]
        return float(self.metric(hypotheses=preds, references=golds).score)


class CorpusLevelPerplexityMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameter for a corpus level perplexity metric.
        Perplexity metrics compute more or less the same thing, which is a variation on the
        average of log-probabilities over a sequence, but the normalization and processing applied
        is different depending on the metric type.
        Perplexity uses an exponential and no weights for the average, weighted perplexity uses an exponential
        and the number of words as weights for the log-prob average, and bits per byte uses the number of bits
        for normalization and divides the results by log(2).

        Args:
            metric_type (str): Can be any of `perplexity`, `weighted_perplexity` or `bits_per_byte`
        """
        if metric_type not in ["perplexity", "weighted_perplexity", "bits_per_byte"]:
            raise ValueError(f"Unknown corpus level perplexity metric type : {metric_type}")

        self.metric_type = metric_type

    def compute(self, items: list[PerplexityCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items."""
        logprobs = [i.logprobs for i in items]
        weights = [i.weights for i in items]

        if self.metric_type == "perplexity":
            return math.exp(-np.mean(logprobs))
        if self.metric_type == "weighted_perplexity":
            return math.exp(-sum(logprobs) / sum(weights))
        if self.metric_type == "bits_per_byte":
            return -sum(logprobs) / sum(weights) * 1 / math.log(2)

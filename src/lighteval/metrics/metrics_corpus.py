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

import logging
import math
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import sacrebleu
import sklearn.metrics

from lighteval.metrics.bayes_at_n import bayes_at_n
from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


class CorpusLevelComputation(ABC):
    @abstractmethod
    def compute_corpus(self, items):
        raise NotImplementedError

    def __str__(self):
        attrs = vars(self)
        attr_strs = []
        for k, v in attrs.items():
            if callable(v):
                val_str = v.__name__
            else:
                val_str = str(v)
            attr_strs.append(f"{k}={val_str}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"


def _is_repeated_full_bayes_prior(non_null_priors: list[object], first_prior: np.ndarray, num_rows: int) -> bool:
    if not all(np.array_equal(np.asarray(prior), first_prior) for prior in non_null_priors):
        return False
    return (first_prior.ndim == 2 and first_prior.shape[0] == num_rows) or (
        first_prior.ndim == 1 and num_rows == 1
    )


def _coerce_bayes_prior_row(prior: object) -> list[int]:
    prior_array = np.asarray(prior)
    if prior_array.ndim == 0:
        raise ValueError("Bayes@N prior rows must be 1D arrays.")
    if prior_array.ndim == 2:
        if prior_array.shape[0] != 1:
            raise ValueError("Bayes@N row-level prior payloads must contain exactly one row.")
        prior_array = prior_array.reshape(-1)
    elif prior_array.ndim != 1:
        raise ValueError("Bayes@N row-level prior payloads must be 1D arrays.")
    return prior_array.tolist()


def _coerce_bayes_prior(priors: list[object | None], num_rows: int) -> list[list[int]] | object | None:
    non_null_priors = [prior for prior in priors if prior is not None]
    if not non_null_priors:
        return None
    if len(non_null_priors) != len(priors):
        raise ValueError("Bayes@N prior observations must be provided for every row or omitted for every row.")

    first_prior = np.asarray(non_null_priors[0])
    if _is_repeated_full_bayes_prior(non_null_priors, first_prior, num_rows):
        return non_null_priors[0]

    prior_rows = [_coerce_bayes_prior_row(prior) for prior in non_null_priors]
    prior_lengths = {len(row) for row in prior_rows}
    if len(prior_lengths) != 1:
        raise ValueError("Bayes@N prior rows must all have the same number of observations.")
    return prior_rows


def _coerce_bayes_items(items: list[dict | list[int]]) -> tuple[list[list[int]], list[float] | None, object | None]:
    if len(items) == 0:
        raise ValueError("Bayes@N needs at least one row.")

    rows = []
    weights = None
    priors = []
    for item in items:
        if isinstance(item, dict):
            if "scores" not in item:
                raise ValueError("Bayes@N payloads must contain a 'scores' row.")
            row = item["scores"]
            item_weights = item.get("weights")
            priors.append(item.get("prior"))
        else:
            row = item
            item_weights = None
            priors.append(None)

        row = list(row)
        if len(row) == 0:
            raise ValueError("Bayes@N rows must contain at least one score.")
        rows.append(row)

        if item_weights is not None:
            item_weights = np.asarray(item_weights, dtype=float)
            if weights is None:
                weights = item_weights
            elif not np.array_equal(weights, item_weights):
                raise ValueError("Bayes@N received inconsistent weights across rows.")

    row_lengths = {len(row) for row in rows}
    if len(row_lengths) != 1:
        raise ValueError("Bayes@N requires every row to have the same number of scores.")

    weights_list = weights.tolist() if weights is not None else None
    return rows, weights_list, _coerce_bayes_prior(priors, len(rows))


class BayesAtNCorpus(CorpusLevelComputation):
    def __init__(self, statistic: Literal["mu", "sigma"]):
        if statistic not in {"mu", "sigma"}:
            raise ValueError("BayesAtNCorpus statistic must be either 'mu' or 'sigma'.")
        self.statistic = statistic

    def compute_corpus(self, items: list[dict | list[int]]) -> float:
        rows, weights, prior = _coerce_bayes_items(items)
        mu, sigma = bayes_at_n(rows, weights=weights, prior=prior)
        return mu if self.statistic == "mu" else sigma


# General aggregations
class MatthewsCorrCoef(CorpusLevelComputation):
    def compute_corpus(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the Matthews Correlation Coefficient, using scikit learn ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)).

        Args:
            items (list[dict]): List of GenerativeCorpusMetricInput

        Returns:
            float: Score
        """
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        return sklearn.metrics.matthews_corrcoef(golds, preds)


class CorpusLevelF1Score(CorpusLevelComputation):
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

    def compute_corpus(self, items: list[LogprobCorpusMetricInput]):
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
            f1s.append(
                sklearn.metrics.f1_score(
                    y_true=[g == i for g in golds], y_pred=[p == i for p in preds], average=self.average
                )
            )
        return float(np.mean(f1s))


class CorpusLevelTranslationMetric(CorpusLevelComputation):
    def __init__(self, metric_type: str, lang: Literal["zh", "ja", "ko", ""] = ""):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
            lang (str): Language code for the translation metric.
        """
        self.metric_type = metric_type
        self.lang = lang

    def get_metric(self):
        if self.metric_type == "bleu":
            import nltk

            nltk.download("punkt_tab")
            return sacrebleu.BLEU(trg_lang=self.lang)
        elif self.metric_type == "chrf":
            return sacrebleu.CHRF()
        elif self.metric_type == "chrf++":
            return sacrebleu.CHRF(word_order=2)
        elif self.metric_type == "ter":
            return sacrebleu.TER(asian_support=True if self.lang != "" else False)
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {self.metric_type}")

    def compute_corpus(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the metric score over all the corpus generated items, by using the sacrebleu implementation."""
        metric = self.get_metric()
        golds = [i.golds for i in items]
        preds = []
        for i in items:
            pred = as_list(i.preds)
            if len(pred) > 1:
                logger.info(
                    f"Multiple predictions present, keeping only the first prediction (when computing sacrebleu.{metric.__name__})."
                )
            preds.append(pred[0])

        if self.metric_type == "bleu":
            golds = [[gold[0] for gold in golds]]

        corpus_score = metric.corpus_score(hypotheses=preds, references=golds)
        score = corpus_score.score
        results = float(score)
        return results


class CorpusLevelPerplexityMetric(CorpusLevelComputation):
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

    def compute_corpus(self, items: list[PerplexityCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items."""
        logprobs = [i.logprobs for i in items]
        weights = [i.weights for i in items]

        if self.metric_type == "perplexity":
            return math.exp(-np.mean(logprobs))
        if self.metric_type == "weighted_perplexity":
            return math.exp(-sum(logprobs) / sum(weights))
        if self.metric_type == "bits_per_byte":
            return -sum(logprobs) / sum(weights) * 1 / math.log(2)

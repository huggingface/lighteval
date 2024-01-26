"""This module manages all the score aggregations and computations occurring at the corpus level.
Some metrics (such as corpus BLEU) are not computed at the individual item level, but over all the corpus.
A number of these aggregations come from the EleutherAIHarness
"""
import math

import numpy as np
import sacrebleu
import sklearn.metrics

from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from lighteval.utils import as_list


# General aggregations
def matthews_corrcoef(items: list[GenerativeCorpusMetricInput]) -> float:
    """Computes the Matthews Correlation Coefficient, using scikit learn ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)).

    Args:
        items (list[dict]): List of the correctly formatted dictionarinput

    Returns:
        float: Score
    """
    golds = [i.golds for i in items]
    preds = [i.preds for i in items]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


class CorpusLevelF1Score:
    def __init__(self, average: str, num_classes: int = 2):
        # If num_classes > 2, we compute multi_f1_corpus_aggregation
        self.average = average  # weighted, macro, micro
        self.num_classes = num_classes

    def compute(self, items):
        golds = [i["golds"] for i in items]
        preds = [i["preds"] for i in items]
        # Single f1
        if self.num_classes == 2:
            fscore = sklearn.metrics.f1_score(golds, preds, average=self.average)
            return np.max(fscore)

        # Multi f1
        f1s = []
        for i in range(self.num_classes):
            f1s.append(sklearn.metrics.f1_score(y_true=golds == i, y_pred=preds == i))
        return np.mean(f1s)


class CorpusLevelTranslationMetric:
    def __init__(self, metric_type: str):
        if metric_type == "bleu":
            self.metric = sacrebleu.corpus_bleu
        elif metric_type == "chrf":
            self.metric = sacrebleu.corpus_chrf
        elif metric_type == "ter":
            self.metric = sacrebleu.corpus_ter
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, items: list[GenerativeCorpusMetricInput]) -> float:
        golds = [i.golds for i in items]
        preds = [as_list(i.preds) for i in items]
        return self.metric(hypotheses=preds, references=golds).score


class CorpusLevelPerplexityMetric:
    def __init__(self, metric_type: str):
        if metric_type not in ["perplexity", "weighted_perplexity", "bits_per_byte"]:
            raise ValueError(f"Unknown corpus level perplexity metric type : {metric_type}")

        self.metric_type = metric_type

    def compute(self, items: list[PerplexityCorpusMetricInput]):
        logprobs = [i.logprobs for i in items]
        weights = [i.weights for i in items]

        if self.metric_type == "perplexity":
            return math.exp(-np.mean(logprobs))
        if self.metric_type == "weighted_perplexity":
            return math.exp(-np.average(logprobs, weights=weights))
        if self.metric_type == "bits_per_byte":
            return -np.average(logprobs, weights=weights) / math.log(2)

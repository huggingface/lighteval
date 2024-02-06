import re
from dataclasses import asdict, dataclass

import numpy as np

from lighteval.logging.hierarchical_logger import hlog_warn


@dataclass
class CorpusMetricInput:
    pass

    def to_dict(self):
        return asdict(self)


@dataclass
class GenerativeCorpusMetricInput(CorpusMetricInput):
    golds: list[str]
    preds: list[str]


@dataclass
class LogprobCorpusMetricInput(CorpusMetricInput):
    golds: list[int]
    preds: list[float]


@dataclass
class PerplexityCorpusMetricInput(CorpusMetricInput):
    logprobs: list[float]
    weights: list[int]


class GenerativePreparator:
    def prepare(golds: list[str], predictions: list[str], **kwargs):
        """Prepares an individual generative example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            golds (list[str]): List of allowed targets for the current example
            predictions (list[str]): List of generated predictions for the current example.

        Returns:
            GenerativeCorpusMetricInput: Stores the golds and predictions as such
        """
        return GenerativeCorpusMetricInput(golds=golds, preds=predictions)


class LoglikelihoodPreparator:
    def __init__(self, is_single_token: bool = False):
        """Init.

        Args:
            is_single_token (bool, optional): True if the preparator is used for single token loglikelihood metrics.
                These metrics are computed faster, as they only compare the single token necessary. Defaults to False.
        """
        self.is_single_token = is_single_token

    def prepare(self, gold_ixs: list[int], choices_logprob: list[float], **kwargs) -> LogprobCorpusMetricInput:
        """Prepares an individual loglikelihood example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            golds_ixs (list[int]): List of the gold indices among the possible choices
            choices_logprob (list[float]): List of each choice's aggregated logprobs (usually with an average or weighted average).

        Returns:
            LogprobCorpusMetricInput: Stores the golds indices and the model's choice (choice with the highest logprob)
                Only the first gold index is taken for a single token loglikelihood metric
        """
        if self.is_single_token:
            if len(gold_ixs) > 1:
                hlog_warn(
                    "The current sample has more than one gold available, which is unexpected. We selected only the first one for the corpus aggregation of the loglikelihood metric."
                )
            return LogprobCorpusMetricInput(golds=gold_ixs[0], preds=np.argmax(choices_logprob))

        return LogprobCorpusMetricInput(golds=gold_ixs, preds=np.argmax(choices_logprob))


class PerplexityPreparator:
    def __init__(self, units_type: str) -> None:
        """Init.

        Args:
            units_type (str): Basic type of text units we want to use to weight perplexity computations.
                Can be `words` or `bytes`

        Raises:
            ValueError: If the unit type is not words or byte, raises a ValueError
        """
        if units_type not in ["words", "bytes"]:
            raise ValueError("Perplexity must be used with either words or bytes.")
        self.units_type = units_type

    def count_units(self, text: str) -> int:
        """Counts the given number of unit in the input text.

        Args:
            text (str): Input text

        Returns:
            int: Number of units of type `self.units_type` in the input text.
        """
        if self.units_type == "words":
            return len(re.split(r"\s+", text))
        if self.units_type == "bytes":
            return len(text.encode("utf-8"))

    def prepare(self, results, reference_text, **kwargs):
        """Prepares an individual perplexity example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            results (list[float]): List of the logprobabilities computed for each item
            reference_text (str): Current reference text for which to compute the length in self.units_type

        Returns:
            PerplexityCorpusMetricInput: Stores the measured logprobs and associated text lengths, counted in the reference unit.
        """
        return PerplexityCorpusMetricInput(logprobs=results.result, weights=self.count_units(reference_text))

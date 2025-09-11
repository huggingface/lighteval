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

import logging
import re
from dataclasses import asdict, dataclass

import numpy as np

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


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


class Preparator:
    pass


class GenerativePreparator(Preparator):
    @staticmethod
    def prepare(doc: Doc, model_response: ModelResponse, **kwargs):
        """Prepares an individual generative example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing predictions.
            **kwargs: Additional keyword arguments.

        Returns:
            GenerativeCorpusMetricInput: Stores the golds and predictions as such
        """
        golds = as_list(doc.get_golds())
        predictions = model_response.final_text
        return GenerativeCorpusMetricInput(golds=golds, preds=predictions)

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


class LoglikelihoodPreparator(Preparator):
    def __init__(self, is_single_token: bool = False):
        """Init.

        Args:
            is_single_token (bool, optional): True if the preparator is used for single token loglikelihood metrics.
                These metrics are computed faster, as they only compare the single token necessary. Defaults to False.
        """
        self.is_single_token = is_single_token

    def prepare(self, doc: Doc, model_response: ModelResponse, **kwargs) -> LogprobCorpusMetricInput:
        """Prepares an individual loglikelihood example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold indices and choices.
            model_response (ModelResponse): The model's response containing logprobs.
            **kwargs: Additional keyword arguments.

        Returns:
            LogprobCorpusMetricInput: Stores the golds indices and the model's choice (choice with the highest logprob)
                Only the first gold index is taken for a single token loglikelihood metric
        """
        gold_ixs = as_list(doc.gold_index)
        choices_logprob = model_response.logprobs
        if self.is_single_token:
            if len(gold_ixs) > 1:
                logger.warning(
                    "The current sample has more than one gold available, which is unexpected. We selected only the first one for the corpus aggregation of the loglikelihood metric."
                )
            return LogprobCorpusMetricInput(golds=gold_ixs[0], preds=np.argmax(choices_logprob))

        return LogprobCorpusMetricInput(golds=gold_ixs, preds=np.argmax(choices_logprob))


class TargetPerplexityPreparator(Preparator):
    def __init__(self, units_type: str) -> None:
        """Init.

        Args:
            units_type (str): Basic type of text units we want to use to weight perplexity computations.
                Can be `words` or `bytes`

        Raises:
            ValueError: If the unit type is not words or byte, raises a ValueError
        """
        if units_type not in ["words", "bytes"]:
            raise ValueError("Perplexity must be computed at either the word or byte level.")
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

    def prepare(self, doc: Doc, model_response: ModelResponse, **kwargs):
        """Prepares an individual perplexity example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing logprobs.
            **kwargs: Additional keyword arguments.

        Returns:
            PerplexityCorpusMetricInput: Stores the measured logprobs and associated text lengths, counted in the reference unit.
        """
        logprobs_flat = np.sum(model_response.logprobs)
        reference_text_flat = " ".join(doc.get_golds())
        return PerplexityCorpusMetricInput(logprobs=logprobs_flat, weights=self.count_units(reference_text_flat))


class PerplexityPreparator(Preparator):
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

    def prepare(self, doc: Doc, model_response: ModelResponse, **kwargs):
        """Prepares an individual perplexity example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing logprobs.
            **kwargs: Additional keyword arguments.

        Returns:
            PerplexityCorpusMetricInput: Stores the measured logprobs and associated text lengths, counted in the reference unit.
        """
        logprobs_flat = np.sum(model_response.logprobs)

        if doc.original_query is not None:
            reference_text_flat = " ".join([doc.original_query])
        else:
            reference_text_flat = " ".join([doc.query])

        return PerplexityCorpusMetricInput(logprobs=logprobs_flat, weights=self.count_units(reference_text_flat))

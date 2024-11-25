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

import re
from functools import lru_cache
from typing import Callable, Literal

import numpy as np

from lighteval.metrics.metrics_sample import (
    ExactMatches,
    F1_score,
    LoglikelihoodAcc,
    NormalizedMultiChoiceProbability,
    Probability,
)
from lighteval.metrics.normalizations import (
    LogProbNormalization,
    LogProbPMINorm,
    LogProbTokenNorm,
    get_multilingual_normalizer,
)
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.utils.formulation import ChoicePrefix, get_prefix
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


def loglikelihood_acc_metric(normalization: LogProbNormalization | None = None) -> SampleLevelMetric:
    """
    Creates a accuracy (loglikelihood) metric, which returns accuracy given normalization.
    """

    normalization_str = normalization.name if normalization else ""
    metric_name = f"acc{'_' + normalization_str if normalization_str else ''}"
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
    metric_name = f"normalized_mc_prob{'_' + normalization_str if normalization_str else ''}"

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
    metric_name = f"prob{'_' + normalization_str if normalization_str else ''}"

    return SampleLevelMetric(
        metric_name=metric_name,
        sample_level_fn=Probability(normalization=normalization, aggregation_function=aggregation_function).compute,
        category=MetricCategory.TARGET_PERPLEXITY,
        use_case=MetricUseCase.PERPLEXITY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def multilingual_quasi_f1_score_metric(
    language: Language, aggregation_function: Callable[[list[float]], float] = max
) -> SampleLevelMetric:
    """
    Creates a language-aware F1 score metric, which returns the F1 score.

    Args:
        language: The language of the samples.
        aggregation_function: Aggregation samples to use when multiple golds are present.

    Returns:
        F1 score metric.
    """
    metric_name = f"f1_{language.value}"

    multilang_normalizer = get_multilingual_normalizer(language)
    return SampleLevelMetric(
        metric_name=metric_name,
        sample_level_fn=F1_score(
            normalize_gold=multilang_normalizer,
            normalize_pred=multilang_normalizer,
            aggregation_function=aggregation_function,
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def multilingual_quasi_exact_match_metric(
    language: Language,
    match_type: Literal["prefix", "suffix", "full"] = "full",
    aggregation_function: Callable[[list[float]], float] = max,
) -> SampleLevelMetric:
    """
    Creates a language-aware exact match metric, which returns the exact match score
    Args:
        language: The language of the samples.
        match_type: The type of match to use
            - "prefix": Prefixes must match
            - "suffix": Suffixes must match
            - "full": Full strings must match
        aggregation_function: Aggregation samples to use when multiple golds are present.
    Returns:
        Exact match metric.
    """
    metric_name = f"exact_match_{language.value}_{match_type}"
    multilang_normalizer = get_multilingual_normalizer(language)
    return SampleLevelMetric(
        metric_name=metric_name,
        sample_level_fn=ExactMatches(
            normalize_gold=multilang_normalizer,
            normalize_pred=multilang_normalizer,
            aggregation_function=aggregation_function,
            type_exact_match=match_type,
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def multilingual_extractive_match_metric(
    language: Language,
    target_for_extraction: Literal["number"] | ChoicePrefix = "number",
    aggregation_function: Callable[[list[float]], float] = max,
) -> SampleLevelMetric:
    # First try to extract the answer from the text
    translation_literal = TRANSLATION_LITERALS[language]

    @lru_cache(maxsize=1)
    def lazy_number_regex():
        number_re = r"(?P<target>\d+(?:\.\d+)?)"
        prefixed_res = [
            f"{translation_literal.answer}.{{0,40}}?{number_re}",
            number_re,
        ]
        return list(map(re.compile, prefixed_res))

    @lru_cache(maxsize=1000)
    def lazy_indices_regex(target_for_extraction: ChoicePrefix, len_gold: int):
        # First get indices to predict
        indices = get_prefix(target_for_extraction, translation_literal)[:len_gold]
        indice_str_re = f"(?P<target>{''.join(indices)})"

        # The answer keys are either surrounded with <space>**answer**., or '<space>answer.' or the same without the dot
        # Same version for comma

        # We try with and without translation literals of punctuation
        full_stop_re = rf"[{translation_literal.full_stop}\.]"
        comma_re = rf"[{translation_literal.comma},]"
        colon_re = rf"[{translation_literal.colon}\:]"
        space_re = rf"[\s{translation_literal.sentence_space}]"

        answer_prefix_re = rf"{space_re}(?:\*\*)?"
        answer_suffix_re = rf"(?:\*\*)?({full_stop_re}|{comma_re}|{colon_re}|{space_re})"
        answer_re = f"{answer_prefix_re}{indice_str_re}{answer_suffix_re}"

        # First we try to extract if by searching for answer followed by a colon, then just answer without any colon, then we just search for answer
        # and if none of this works, we just search for the indices in the text
        answer_word = translation_literal.answer

        prefixed_res = [
            f"{answer_word}.{{0,40}}?{colon_re}{answer_re}",
            f"{answer_word}.{{0,40}}?{answer_re}",
            answer_re,
            indice_str_re,
        ]
        return list(map(re.compile, prefixed_res))

    def evaluate_one_item(gold: str, pred: str, target_re: list[re.Pattern]) -> float:
        for re_pattern in target_re:
            matches = re_pattern.findall(pred)
            if matches:
                return 1 if matches[-1].strip() == gold.strip() else 0

        return 0

    def extract_target(
        target_for_extraction: Literal["number"] | ChoicePrefix,
        golds: list[str],
        predictions: list[str],
        formatted_doc: Doc,
    ) -> float:
        if target_for_extraction == "number":
            target_re = lazy_number_regex()
        else:
            target_re = lazy_indices_regex(target_for_extraction, len(formatted_doc.choices))

        results = [evaluate_one_item(gold, pred, target_re) for gold, pred in zip(golds, predictions)]
        return aggregation_function(results)

    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=extract_target,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

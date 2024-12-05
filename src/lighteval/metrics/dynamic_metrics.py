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
from itertools import product
from typing import Callable, Literal

import numpy as np
import sympy

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
    math_normalizer,
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
    target_for_extraction: list[Literal["number", "latex"] | ChoicePrefix] = ["number"],
    aggregation_function: Callable[[list[float]], float] = max,
) -> SampleLevelMetric:
    translation_literal = TRANSLATION_LITERALS[language]

    @lru_cache(maxsize=1)
    def lazy_number_regex():
        # Basic number patterns (no LaTeX)
        number_re = (
            r"(?P<target>"
            r"\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d+)?|"  # Numbers with thousand/decimal separators
            r"\d+(?:[.,]\d+)?"  # Simple numbers with decimals
            r")"
        )

        # Match after equals with answer word
        equals_re = f"(?i:{translation_literal.answer}).{{0,40}}?=\\s*{number_re}"

        # Match with answer word
        answer_re = f"(?i:{translation_literal.answer}).{{0,40}}?{number_re}"

        # Match plain numbers
        plain_number_re = number_re

        return [re.compile(pattern) for pattern in [equals_re, answer_re, plain_number_re]]

    @lru_cache(maxsize=1)
    def lazy_latex_regex():
        # Only LaTeX expressions between delimiters
        latex_re = (
            r"(?P<target>"
            r"\$\$[\s\S]*?\$\$|"  # $$...$$ (display math, can be multiline)
            r"\\\[[\s\S]*?\\\]|"  # \[...\] (display math, can be multiline)
            r"\$[^\n$]*?\$|"  # $...$ (inline math, single line)
            r"\\\([^\n)]*?\\\)"  # \(...\) (inline math, single line)
            r")"
        )

        # Match after equals with answer word
        equals_re = f"(?i:{translation_literal.answer}).{{0,40}}?=\\s*{latex_re}"

        # Match with answer word
        answer_re = f"(?i:{translation_literal.answer}).{{0,40}}?{latex_re}"

        # Match plain LaTeX
        plain_latex_re = latex_re

        return [re.compile(pattern, re.DOTALL) for pattern in [equals_re, answer_re, plain_latex_re]]

    @lru_cache(maxsize=1000)
    def lazy_indices_regex(target_for_extraction: ChoicePrefix, len_choices: int):
        # First get indices to predict
        indices = get_prefix(target_for_extraction, translation_literal)[:len_choices]
        indice_str_re = f"(?P<target>[{''.join([re.escape(i) for i in indices])}])"

        # The answer keys are either surrounded with <space>**answer**., or '<space>answer.' or the same without the dot
        full_stop_re = rf"[{re.escape(translation_literal.full_stop)}\.]"
        comma_re = rf"[{re.escape(translation_literal.comma)}\,]"
        colon_re = rf"[{re.escape(translation_literal.colon)}\:]"
        space_re = rf"(?:\s|{re.escape(translation_literal.sentence_space)})"

        answer_prefix_re = rf"{space_re}(?:\*\*)?"
        answer_suffix_re = rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|$)"
        answer_re = f"{answer_prefix_re}{indice_str_re}{answer_suffix_re}"
        answer_re_start = rf"^(?:\*\*)?{indice_str_re}{answer_suffix_re}"

        answer_word = f"(?i:{translation_literal.answer})"

        prefixed_res = [
            # Answer is: A.
            f"{answer_word}.{{0,40}}?{colon_re}{answer_re}",
            # Answer is A.
            f"{answer_word}.{{0,40}}?{answer_re}",
            # A. at start
            answer_re_start,
            # A.
            answer_re,
            # A
            indice_str_re,
        ]
        return list(map(re.compile, prefixed_res))

    def extract_target_from_pred(pred: str, target_re: list[re.Pattern]) -> str | None:
        for re_pattern in target_re:
            matches = re_pattern.findall(pred)
            if matches:
                match = matches[-1]
                return match
        return None

    def extract_math(match: str, target_type: str) -> str | None:
        """Extract numerical value from a match.

        Args:
            match: The matched string (either LaTeX or number)
            target_type: Either "latex" or "number"

        Returns:
            Numerical value as string or None if parsing fails
        """
        try:
            if target_type == "latex":
                # Use math_normalizer to handle LaTeX
                normalized = math_normalizer(match)
                if normalized:
                    # math_normalizer already converts to a sympy-parseable format
                    result = sympy.sympify(normalized).evalf()
                    return str(float(result))
            else:  # number
                # Clean up the number (remove spaces, normalize comma/period)
                clean_num = match.replace(". ", "")
                # Also use math_normalizer here for consistency
                normalized = math_normalizer(clean_num)
                if normalized:
                    result = sympy.sympify(normalized).evalf()
                    return str(float(result))
        except:
            return None

    def extract_target(
        golds: list[str],
        predictions: list[str],
        formatted_doc: Doc,
    ) -> float:
        # Try each target type in order
        for target_type in target_for_extraction:
            if target_type == "number":
                target_re = lazy_number_regex()
            elif target_type == "latex":
                target_re = lazy_latex_regex()
            else:
                target_re = lazy_indices_regex(target_type, len(formatted_doc.choices))

            extracted_predictions = []
            for pred in predictions:
                match = extract_target_from_pred(pred, target_re)
                if match:
                    if target_type in ["number", "latex"]:
                        value = extract_math(match, target_type)
                        if value is not None:
                            extracted_predictions.append(value)
                    else:
                        extracted_predictions.append(match)

            if extracted_predictions:  # If we found matches, process them
                results = []
                for gold, extracted_pred in product(golds, extracted_predictions):
                    if target_type in ["number", "latex"]:
                        try:
                            # Normalize gold value too
                            gold_normalized = math_normalizer(gold)
                            if not gold_normalized:
                                continue

                            gold_val = float(sympy.sympify(gold_normalized).evalf())
                            pred_val = float(sympy.sympify(extracted_pred).evalf())
                            results.append(1 if abs(gold_val - pred_val) < 1e-6 else 0)
                        except:
                            # Fall back to string comparison
                            if gold and extracted_pred:
                                results.append(1 if gold.strip() == extracted_pred.strip() else 0)
                            else:
                                results.append(0)
                    else:
                        if gold and extracted_pred:
                            results.append(1 if gold.strip() == extracted_pred.strip() else 0)
                        else:
                            results.append(0)

                return aggregation_function(results or [0])

        return 0.0  # Return 0 if no matches found with any target type

    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=extract_target,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

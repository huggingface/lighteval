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
from typing import Callable, Literal, Tuple

import numpy as np
import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

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
    target_for_extraction: list[Literal["latex", "expr"] | ChoicePrefix] = ["expr"],
    extract_all_targets: bool = False,
    aggregation_function: Callable[[list[float]], float] = max,
) -> SampleLevelMetric:
    translation_literal = TRANSLATION_LITERALS[language]

    # @lru_cache(maxsize=1)
    # def lazy_expr_regex():
    #     # Operators (+-*/^()÷)
    #     # Make sure the expression contains at least one operator, so that we don't just match numbers
    #     equals_expr_re = rf"(?i:{translation_literal.answer}|{translation_literal.result_word})(?:.{{0,100}}=\s*|.{{0,50}}?)({expr_re})(?!\s*=)"
    #     return [re.compile(pattern) for pattern in [equals_expr_re, expr_re]]

    @lru_cache(maxsize=1)
    def lazy_expr_regex():
        # Basic number patterns (no LaTeX)
        number_re = (
            # Format 1: Numbers with thousand separators (e.g., "1,234.56" or "1 234.56")
            r"(?P<integer1>-?\d{1,3}(?:[ ,]\d{3})+)(?P<decimal1>\.\d+)?|"
            # Format 2: Simple numbers with decimal point or comma (e.g., "123.45" or "123,45")
            r"(?P<integer2>-?\d+)(?P<decimal2>[.,]\d+)|"
            # Format 3: Decimal part only (e.g., ".123")
            r"(?P<decimal3>\.\d+)|"
            # Format 4: Integer only (e.g., "123")
            r"(?P<integer3>-?\d+)"
        )

        operators = [r"\+", r"\-", r"\*", r"\×", r"\/", r"\^", r"\(", r"\)", r"\÷"]
        operators_re = "".join(operators)
        all_expr_chars = r"[\d\.\s" + operators_re + r"]"
        # Expression should have at minimum at least one operator, must start with a digit
        expr_re = rf"-?\(?-?\d{all_expr_chars}*[{operators_re}]{all_expr_chars}+\)?"
        colon_re = rf"[{re.escape(translation_literal.colon)}\:]"

        # Match after the last equals with answer word - require the number pattern
        equals_re_colon = rf"(?i:{translation_literal.answer}|{translation_literal.result_word}){colon_re}(?:.{{0,100}}=\s*|.{{0,50}}?)(?P<expr>{number_re}|{expr_re})(?!\s*=)"
        equals_re = rf"(?i:{translation_literal.answer}|{translation_literal.result_word})(?:.{{0,100}}=\s*|.{{0,50}}?)(?P<expr>{number_re}|{expr_re})(?!\s*=)"

        # We first try to match the answer then the plain number
        return [
            re.compile(pattern)
            for pattern in [equals_re_colon, equals_re, f"(?P<expr>{expr_re})", f"(?P<expr>{number_re})"]
        ]

    @lru_cache(maxsize=1)
    def lazy_latex_regex():
        # Only LaTeX expressions between delimiters
        simple_number = r"-?\d+(?:[.,]\d+)?"
        latex_re = (
            r"("
            r"(?<!\\)\$\$(?P<latex1>[\s\S]+?)(?<!\\)\$\$|"  # $$...$$ (display math, can be multiline)
            r"(?<!\\)\\\[(?P<latex2>[\s\S]+?)(?<!\\)\\\]|"  # \[...\] (display math, can be multiline)
            r"(?<!\\)\$(?P<latex3>(?:\\[$]|[^\n$])+?)(?<!\\)\$|"  # $...$ (inline math, single line, allows escaped $)
            r"(?<!\\)\\\((?P<latex4>[^\n)]+?)(?<!\\)\\\)|"  # \(...\) (inline math, single line)
            r"(?<!\\)\[(?P<latex5>[^\n$]+?)(?<!\\)\]|"  # While this is no a valid display math llms like to generate it, allow it
            rf"(?P<latex6>-?\\frac{{{simple_number}}}{{{simple_number}}})"  # Simple fraction without any signaling
            r")"
        )
        colon_re = rf"[{re.escape(translation_literal.colon)}\:]"

        # Match with answer word
        answer_re_colon = (
            f"(?i:{translation_literal.answer}|{translation_literal.result_word}){colon_re}.{{0,100}}?{latex_re}"
        )
        answer_re = f"(?i:{translation_literal.answer}|{translation_literal.result_word}).{{0,100}}?{latex_re}"

        # Match plain LaTeX
        plain_latex_re = f"{latex_re}"

        return [re.compile(pattern, re.DOTALL) for pattern in [answer_re_colon, answer_re, plain_latex_re]]

    @lru_cache(maxsize=1000)
    def lazy_indices_regex(target_for_extraction: ChoicePrefix, len_choices: int):
        # First get indices to predict
        indices = get_prefix(target_for_extraction, translation_literal)[:len_choices]
        indice_str_re = f"(?P<indices>{'|'.join([re.escape(i) for i in indices])})"

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
            f"{answer_word}{colon_re}.{{0,50}}?{answer_re}",
            # Answer is A.
            f"{answer_word}.{{0,50}}?{answer_re}",
            # A. at start
            answer_re_start,
            # A.
            answer_re,
            # A
            indice_str_re,
        ]
        return list(map(re.compile, prefixed_res))

    def extract_expr(match: re.Match) -> tuple[str | None, sympy.Expr | None]:
        # First combine the number
        groups = match.groupdict()
        # This musst always exist
        expr = groups["expr"]
        integer = next((val for name, val in groups.items() if name.startswith("integer") and val), None)
        decimal = next((val for name, val in groups.items() if name.startswith("decimal") and val), None)

        if integer:
            # Remove thousand separators and convert to float
            num_str = integer.translate(str.maketrans("", "", ", "))

            if decimal:
                # Add decimal part if present
                num_str += f"{decimal.replace(',', '.')}"
            return None, sympy.Number(float(num_str))
        elif decimal:
            # Just decimal part, convert to float
            return None, sympy.Number(float(f"0{decimal}"))

        # Otherwise just return the expression
        # Remove new lines and spaces
        expr = expr.replace("\n", "").replace(" ", "")
        try:
            return expr, parse_expr(expr)
        except:
            return expr, None

    def extract_latex(match: re.Match) -> tuple[str | None, sympy.Expr | None]:
        # Take last expr after the =
        latex = next((val for name, val in match.groupdict().items() if name.startswith("latex") and val))
        latex = re.split(r"(?<!<|>)=", latex)[-1]  # Split on = not preceded by < or >
        # Remove whitespace and new lines and tabs
        latex = latex.replace("\n", "").replace("\t", "")
        normalized_latex = math_normalizer(f"\\boxed{{{latex}}}")
        try:
            return None, parse_latex(normalized_latex)
        except:
            return normalized_latex, None

    def extract_match(match: re.Match, target_type: str) -> tuple[str | None, str | sympy.Expr | float | None]:
        if target_type == "latex":
            return extract_latex(match)
        elif target_type == "expr":
            return extract_expr(match)

        elif target_type == "indices":
            return match.group(0), match.group("indices")

        return match.group(0), None

    def extract_target_from_pred(
        pred: str, target_res: list[Tuple[list[re.Pattern], str]]
    ) -> list[str | sympy.Expr | None | float]:
        fallbacks = [None for _ in target_res]
        extracted_predictions = [None for _ in target_res]

        for i, (patterns, t) in enumerate(target_res):
            for p in patterns:
                re_pattern, target_type = p, t
                matches = list(re_pattern.finditer(pred))
                fallback, extracted_match = extract_match(matches[-1], target_type) if matches else (None, None)

                # If we don't have feedback yet, use the fallback as long as it's not empty or None
                if fallbacks[i] is None and fallback:
                    fallbacks[i] = fallback

                # If we managed to extract something, break
                if extracted_match is not None:
                    extracted_predictions[i] = extracted_match
                    break

            # Break early if we don't want to extract all targets
            if not extract_all_targets and any(e is not None for e in extracted_predictions):
                break

        # First filter out the None values
        extracted_predictions = [e for e in extracted_predictions if e is not None]

        # If we don't have any predictions, use the fallback
        if not any(e is not None for e in extracted_predictions):
            extracted_predictions = [f for f in fallbacks if f is not None]

        return extracted_predictions

    def compare_gold_target(gold: list[str | sympy.Expr | float], target: list[str | sympy.Expr | float]) -> float:
        def compare_single_extraction(gold: str | sympy.Expr | float, target: str | sympy.Expr | float) -> float:
            # Expression case
            if isinstance(gold, sympy.Expr) and isinstance(target, sympy.Expr):
                # First try using -
                simplified = sympy.simplify(gold - target)
                if simplified.is_zero:
                    return 1.0
                # Otherwise try using ==
                elif gold == target:
                    return 1.0
                # Just use str comparison
                else:
                    return 1.0 if str(gold.evalf()) == str(target.evalf()) else 0.0

            elif isinstance(gold, str) or isinstance(target, str):
                gold = str(gold)
                target = str(target)
                # Ensure it's both not empty and equal
                return len(gold) > 0 and len(target) > 0 and gold == target
            return 0.0

        return any(compare_single_extraction(g, t) for g, t in product(gold, target))

    def extract_target(
        golds: list[str],
        predictions: list[str],
        formatted_doc: Doc,
    ) -> float:
        # Try each target type in order
        extraction_res = [
            (lazy_latex_regex(), "latex")
            if target_type == "latex"
            else (lazy_expr_regex(), "expr")
            if target_type == "expr"
            else (lazy_indices_regex(target_type, len(formatted_doc.choices)), "indices")
            for target_type in target_for_extraction
        ]

        # Sort the extraction res so that order is indices, latex, expr
        extraction_res = sorted(extraction_res, key=lambda x: {"indices": 0, "latex": 1, "expr": 2}.get(x[1], 3))

        extracted_predictions = [extract_target_from_pred(pred, extraction_res) for pred in predictions]
        extracted_golds = [extract_target_from_pred(gold, extraction_res) for gold in golds]

        return aggregation_function(
            (1.0 if any(compare_gold_target(gold, pred) for gold in extracted_golds) else 0.0)
            for pred in extracted_predictions
        )

    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=extract_target,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

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
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import product
from typing import Callable, Literal, Sequence

import numpy as np
import sympy
from latex2sympy2 import latex2sympy as parse_latex
from sympy import FiniteSet, Interval
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import parse_expr
from sympy.matrices import MatrixBase

from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.metrics.metrics_sample import (
    ExactMatches,
    F1_score,
    JudgeLLMMixEval,
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
    remove_outer_braces,
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


@dataclass(frozen=True)
class LatexExtractionConfig:
    """
    Config for extracting latex from the prediction.
    Args:
        groups_with_fallback: Groups, which will return a fallback value (postprocessed latex) if latex matching failed.
    """

    groups_with_fallback: Sequence[str] = ("latexDisplayDollar", "latexDisplayBracket", "latexInlineParenthesis")
    try_last_latex_match: bool = True


@dataclass(frozen=True)
class ExprExtractionConfig:
    try_last_expr_match: bool = True


@dataclass(frozen=True)
class IndicesExtractionConfig:
    prefix_for_extraction: ChoicePrefix
    try_last_indices_match: bool = True


ExtractionTarget = LatexExtractionConfig | ExprExtractionConfig | IndicesExtractionConfig


def try_parse_latex_interval(latex: str) -> Interval | FiniteSet | None:
    # TODO: Move to antlr in future
    # Simply check if the latex is an interval -> [/(\number, \number)/]
    match = re.match(r"(?P<l_bound>[\(\[])\s*(?P<l_val>.*?),\s*(?P<u_val>.*?)(?P<u_bound>[\)\]])", latex.strip())
    if match:
        l_closed = match.group("l_bound") == "["
        u_closed = match.group("u_bound") == "]"

        try:
            l_bound_val = parse_latex(match.group("l_val"))
            # Ensures we are not parsing a set
            if isinstance(l_bound_val, list):
                return None
            u_bound_val = parse_latex(match.group("u_val"))
            if isinstance(u_bound_val, list):
                return None
            # Ensure that the result is not list
            interval = Interval(l_bound_val, u_bound_val, l_closed, u_closed)
            # If the interval is empty, there is probably issue with sides so just switch them
            if interval.is_empty:
                interval = Interval(u_bound_val, l_bound_val, u_closed, l_closed)
            return interval
        except:
            return None

    return None

pm_regex = re.compile(r"(?P<expr>.+)(?P<plus_minus>\\pm)(?P<value>.+)")
def try_parse_latex_set(latex: str) -> FiniteSet | None:
    elements = []
    latex = remove_outer_braces(latex)
    for latex_element in re.split(r"(?<!\\),", latex.strip()):
        match = pm_regex.match(latex_element.strip())
        if match:
            try:
                expr = parse_latex(match.group("expr"))
                value = parse_latex(match.group("value"))
                # Return a tuple of (expr - value, expr + value)
                elements.append(expr - value)
                elements.append(expr + value)
            except:
                return None
        else:
            try:
                elements.append(parse_latex(latex_element.strip()))
            except:
                return None
    if len(elements) == 1:
        return elements[0]

    return FiniteSet(*elements)

def parse_latex_extended(latex: str) -> FiniteSet | Interval | sympy.Expr | None:
    interval = try_parse_latex_interval(latex)
    if interval is not None:
        return interval
    return try_parse_latex_set(latex)

def extract_expr(match: re.Match) -> tuple[str | sympy.Expr | None, str]:
    # First combine the number
    groups = match.groupdict()
    # This musst always exist
    expr = groups["expr"]
    integer = next((val for name, val in groups.items() if name.startswith("integer") and val), None)
    decimal = next((val for name, val in groups.items() if name.startswith("decimal") and val), None)

    percentage_multiplier = 0.01 if groups.get("percent", None) else 1

    if integer:
        # Remove thousand separators and convert to float
        num_str = integer.translate(str.maketrans("", "", ", "))

        if decimal:
            # Add decimal part if present
            num_str += f"{decimal.replace(',', '.')}"
        return sympy.Number(float(num_str)) * percentage_multiplier, expr
    elif decimal:
        # Just decimal part, convert to float
        return sympy.Number(float(f"0{decimal}")) * percentage_multiplier, expr

    # Otherwise just return the expression
    # Remove new lines and spaces
    try:
        return parse_expr(expr.replace("\n", "").replace(" ", "")), expr
    except:
        return None, expr


def extract_latex(match: re.Match, target_type: LatexExtractionConfig) -> tuple[sympy.Expr | str | None, str]:
    latex_group, latex = next(
        ((name, val) for name, val in match.groupdict().items() if name.startswith("latex") and val), ("", "")
    )

    normalized_latex = math_normalizer(latex)

    parsed_latex = parse_latex_extended(normalized_latex)
    if isinstance(parsed_latex, list):
        return FiniteSet(*parsed_latex), normalized_latex
    return parsed_latex, normalized_latex

    # If we got this catch by hard to see in wild latex expression, it's possibly that it's a false positive, otherwise we probably failed because the latex is not valid
    # return (
    #     normalized_latex
    #     if len(normalized_latex.strip()) > 0 and latex_group in target_type.groups_with_fallback
    #     else None
    # )
    return None, normalized_latex


def extract_match(match: re.Match, target_type: ExtractionTarget) -> tuple[str | sympy.Expr | float | None, str]:
    """
    Extracts the match from the regex match.
    Returns a tuple of the extracted value and string representation of the match
    """
    if isinstance(target_type, LatexExtractionConfig):
        return extract_latex(match, target_type)
    elif isinstance(target_type, ExprExtractionConfig):
        return extract_expr(match)
    elif isinstance(target_type, IndicesExtractionConfig):
        return match.group("indices"), match.group("indices")


@lru_cache(maxsize=1)
def lazy_expr_regex(expr_config: ExprExtractionConfig, language: Language) -> list[re.Pattern[str]]:
    translation_literal = TRANSLATION_LITERALS[language]

    # TODO: Possibly we should also nesure that the expression doesn't appear in latex env
    # Basic number patterns (no LaTeX)
    number_re = (
        # Format 1: Numbers with thousand separators (e.g., "1,234.56" or "1 234.56")
        r"(?:"
        r"(?P<integer1>-?\d{1,3}(?:[ ,]\d{3})+)(?P<decimal1>\.\d+)?|"
        # Format 2: Simple numbers with decimal point or comma (e.g., "123.45" or "123,45")
        r"(?P<integer2>-?\d+)(?P<decimal2>[.,]\d+)|"
        # Format 3: Decimal part only (e.g., ".123")
        r"(?P<decimal3>\.\d+)|"
        # Format 4: Integer only (e.g., "123")
        r"(?P<integer3>-?\d+)"
        r")(?P<percent>%?|%)"
    )

    operators = [r"\+", r"\-", r"\*", r"\×", r"\/", r"\^", r"\(", r"\)", r"\÷"]
    operators_re = "".join(operators)
    all_expr_chars = r"[\d\.\s" + operators_re + r"]"
    # Expression should have at minimum at least one operator, must start with a digit
    expr_re = rf"-?\(?-?\d{all_expr_chars}*[{operators_re}]{all_expr_chars}+\)?"

    # Punctuation regexes
    full_stop_re = rf"[{re.escape(translation_literal.full_stop)}\.]"
    comma_re = rf"[{re.escape(translation_literal.comma)}\,]"
    colon_re = rf"[{re.escape(translation_literal.colon)}\:]"
    space_re = rf"(?:\s|{re.escape(translation_literal.sentence_space)})"

    # For expressions we also allow = prefix without any space, for suffix we allow ) because sometimes the answer is wrapped in parenthesis
    expr_prefix_re = rf"(?:^|{space_re}|\=)(?:\*\*)?"
    expr_suffix_re = rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|\)|\$|$)"

    expr = f"(?P<expr>{expr_re}|{number_re})"
    full_expr = rf"(?:{expr_prefix_re}{expr}{expr_suffix_re})"
    regexes: list[str] = []
    if language == Language.ENGLISH:
        equals_re = rf"(?i:the final answer is\s*){full_expr}"

    answer_prefix_re = rf"(?i:{translation_literal.answer}|{translation_literal.result_word})"
    # Match after the last equals with answer word - require the number pattern
    # Not sure about the equals matchings

    equals_re_colon = rf"{answer_prefix_re}{colon_re}(?:.{{0,100}}=\s*|.{{0,50}}?){full_expr}(?!\s*=)"
    equals_re = rf"{answer_prefix_re}(?:.{{0,100}}=\s*|.{{0,50}}?){full_expr}(?!\s*=)"

    regexes.extend([equals_re_colon, equals_re])
    if expr_config.try_last_expr_match:
        regexes.append(f"({expr_prefix_re})(?P<expr>{expr_re})({expr_suffix_re})")
        regexes.append(f"({expr_prefix_re})(?P<expr>{number_re})({expr_suffix_re})")

        # We never try to match without prefixes as otherwise it will easily match partials

    # We first try to match the answer then the plain number
    return [re.compile(pattern) for pattern in regexes]


@lru_cache(maxsize=1)
def lazy_latex_regex(latex_config: LatexExtractionConfig, language: Language):
    # Only LaTeX expressions between delimiters
    simple_number = r"-?\d+(?:[.,]\d+)?"
    latex_envs_re = (
        r"("
        r"(?<!\\)\$\$(?P<latexDisplayDollar>[\s\S]+?)(?<!\\)\$\$|"  # $$...$$ (display math, can be multiline)
        r"(?<!\\)\\\[(?P<latexDisplayBracket>[\s\S]+?)(?<!\\)\\\]|"  # \[...\] (display math, can be multiline)
        r"(?<!\\|\d)\$(?P<latexInlineDollar>(?:\\[$]|[^\n$])+?)(?<!\\)\$|"  # $...$ (inline math, single line, allows escaped $), we make sure it's not preceed by a digit to minimize false positives with actualy dollar unit
        r"(?<!\\)\\\((?P<latexInlineParenthesis>[^\n)]+?)(?<!\\)\\\)|"  # \(...\) (inline math, single line)
        r"(?<!\\)\[(?P<latexInlineBracket>[^\n$]+?)(?<!\\)\]"  # [....] While this is no a valid display math llms like to generate it, allow it
        r")"
    )

    # Match latex without environments
    latex_boxed = r"(?P<latexBoxed>\\boxed{[^\n]+})"  # Boxed number, it's fine to be as greedy as possible as we will find the correct end afterwards
    latex_fraction = rf"(?P<latexFraction>-?\\frac{{{simple_number}}}{{{simple_number}}})"

    translation_literal = TRANSLATION_LITERALS[language]
    colon_re = rf"[{re.escape(translation_literal.colon)}\:]"

    answer_prefix_re = rf"(?i:{translation_literal.answer}|{translation_literal.result_word})"

    # We first match boxed env, for some reason that's the most common case of output
    # Then we match the latex with environments, then we try to match the fraction
    regexes: list[str] = []
    for latex_re in [latex_boxed, latex_envs_re, latex_fraction]:
        if language == Language.ENGLISH:
            custom_answer_re = rf"final answer is\s*{latex_re}"
            regexes.append(custom_answer_re)

        # Match with answer word
        answer_re_colon = f"{answer_prefix_re}{colon_re}.{{0,50}}?{latex_re}"
        answer_re = f"{answer_prefix_re}.{{0,50}}?{latex_re}"

        regexes.extend([answer_re_colon, answer_re])

        # Match plain LaTeX
        if latex_config.try_last_latex_match:
            regexes.append(latex_re)

    return [re.compile(pattern, re.DOTALL) for pattern in regexes]


@lru_cache(maxsize=100)
def lazy_indices_regex(indices_config: IndicesExtractionConfig, len_choices: int, language: Language):
    translation_literal = TRANSLATION_LITERALS[language]
    # First get indices to predict
    indices = get_prefix(indices_config.prefix_for_extraction, translation_literal)[:len_choices]
    indice_str_re = f"(?P<indices>{'|'.join([re.escape(i) for i in indices])})"

    # The answer keys are either surrounded with <space>**answer**., or '<space>answer.' or the same without the dot
    full_stop_re = rf"[{re.escape(translation_literal.full_stop)}\.]"
    comma_re = rf"[{re.escape(translation_literal.comma)}\,]"
    colon_re = rf"[{re.escape(translation_literal.colon)}\:]"
    space_re = rf"(?:\s|{re.escape(translation_literal.sentence_space)})"

    answer_prefix_re = rf"(^|{space_re})(?:\*\*)?"
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
    ]
    if indices_config.try_last_indices_match:
        prefixed_res.extend(
            [
                # A.
                answer_re,
                # A
                indice_str_re,
            ]
        )

    return list(map(re.compile, prefixed_res))


def get_extraction_regexes(
    formatted_doc: Doc, target_types: tuple[ExtractionTarget], language: Language
) -> list[tuple[list[re.Pattern], ExtractionTarget]]:
    extraction_regexes = [
        (lazy_latex_regex(target_type, language), target_type)
        if isinstance(target_type, LatexExtractionConfig)
        else (lazy_expr_regex(target_type, language), target_type)
        if isinstance(target_type, ExprExtractionConfig)
        else (lazy_indices_regex(target_type, len(formatted_doc.choices), language), target_type)
        for target_type in target_types
    ]

    # Sort the extraction res so that order is indices, latex, expr
    def get_target_type_order(target_type: ExtractionTarget) -> int:
        match target_type:
            case IndicesExtractionConfig():
                return 0
            case LatexExtractionConfig():
                return 1
            case ExprExtractionConfig():
                return 2

    extraction_regexes = sorted(extraction_regexes, key=lambda x: get_target_type_order(x[1]))

    return extraction_regexes


def extract_target_from_pred(
    pred: str, 
    target_res: list[tuple[list[re.Pattern], ExtractionTarget]], 
    extraction_mode: Literal["first_match", "extract_each_target", "first_fallback"] = "first_match",
    fallback_mode: Literal["no_fallback", "first_match", "any_match"] = "no_fallback"
    
) -> list[str | sympy.Expr | None | float]:
    extracted_predictions = []
    fallbacks = []

    for patterns, target_type in target_res:
        for p in patterns:
            matches = list(p.finditer(pred))
            extracted_match, str_fallback = extract_match(matches[-1], target_type) if matches else (None, None)

            # If we managed to extract something, break
            if extracted_match is not None:
                extracted_predictions.append(extracted_match)
                break

            if str_fallback:
                fallbacks.append(str_fallback)
        # Break early if we don't want to extract all targets
        if extraction_mode == "first_match" and extracted_predictions:
            break

        if extraction_mode == "first_fallback" and fallbacks:
            break

    # Handle fallback modes
    if not extracted_predictions:  # Only use fallbacks if no successful extractions
        if fallback_mode == "first_match" and fallbacks:
            return [fallbacks[0]]  # Return first fallback
        elif fallback_mode == "any_match" and fallbacks:
            return fallbacks  # Return all fallbacks
        elif fallback_mode == "no_fallback":
            return []  # Return empty list if no successful extractions
    
    return extracted_predictions

def sympy_expr_eq(a: sympy.Expr | MatrixBase, b: sympy.Expr | MatrixBase, precision: int) -> bool:
    # This should be enough for most cases
    a_b_diff = sympy.simplify(a - b)
    if isinstance(a_b_diff, MatrixBase) and a_b_diff.is_zero_matrix:
        return True
    elif isinstance(a_b_diff, sympy.Expr) and a_b_diff.is_zero:
        return True

    return try_evalf(a, precision) == try_evalf(b, precision)

def try_evalf(a: sympy.Expr | MatrixBase, precision: int) -> str:
    try:
        return str(a.evalf(n=precision))
    except:
        return str(a)

<<<<<<< HEAD

def compare_gold_target(gold: list[sympy.Expr | Relational], target: list[sympy.Expr | Relational]) -> float:
    def compare_sympy_expr(gold: sympy.Expr | Relational , target: sympy.Expr | Relational) -> float:
        if (isinstance(gold, (sympy.Expr, MatrixBase))) and (isinstance(target, (sympy.Expr, MatrixBase))):
            try:
                return 1.0 if sympy_expr_eq(gold, target) else 0.0
            except:
                return 0.0
=======
def compare_gold_target(gold: list[str | sympy.Expr | float], target: list[str | sympy.Expr | float], precision: int) -> float:
    def compare_single_extraction(gold: str | sympy.Expr | float, target: str | sympy.Expr | float) -> float:
        # Expression case
        if (isinstance(gold, (sympy.Expr, MatrixBase))) and (isinstance(target, (sympy.Expr, MatrixBase))):
            return 1.0 if sympy_expr_eq(gold, target, precision) else 0.0
>>>>>>> 3ad0971 (tmp)

        # Support for equations
        elif isinstance(gold, Relational) and isinstance(target, Relational):
            # Helper to check if expressions are equivalent when flipped
            def are_flipped_inequalities_equal(a: Relational, b: Relational) -> bool:
                return sympy_expr_eq(a.lhs - a.rhs, b.rhs - b.lhs, precision)

            # Same type of relation (e.g. both <= or both >=)
            if type(gold) == type(target) and sympy_expr_eq(gold.lhs - gold.rhs, target.lhs - target.rhs, precision):
                return 1.0

            # Check flipped inequalities (a <= b equals b >= a)
            if (
                isinstance(gold, sympy.GreaterThan)
                and isinstance(target, sympy.LessThan)
                or isinstance(gold, sympy.LessThan)
                and isinstance(target, sympy.GreaterThan)
            ) and are_flipped_inequalities_equal(gold, target):
                return 1.0

            return 0.0

        elif (
            (isinstance(gold, (Interval, FiniteSet)) and isinstance(target, (Interval, FiniteSet)))
            and gold.symmetric_difference(target).is_empty
        ):
            return 1.0
        
        return 0.0



    def compare_single_extraction(gold: str | sympy.Expr | float, target: str | sympy.Expr | float) -> float:
        # Expression case

        if isinstance(gold, sympy.Basic) and isinstance(target, sympy.Basic):
            try:
                return 1.0 if compare_sympy_expr(gold, target) else 0.0
            except:
                pass

        # We just do string comparison for everything else
<<<<<<< HEAD
        gold = try_evalf(gold) if isinstance(gold, sympy.Expr) else str(gold)
        target = try_evalf(target) if isinstance(target, sympy.Expr) else str(target)
=======
        else:
            gold = try_evalf(gold, precision) if isinstance(gold, sympy.Expr) else str(gold)
            target = try_evalf(target, precision) if isinstance(target, sympy.Expr) else str(target)
>>>>>>> 3ad0971 (tmp)

        gold = gold.strip()
        target = target.strip()

        # Ensure it's both not empty and equal
        return len(gold) > 0 and len(target) > 0 and gold == target


    return any(compare_single_extraction(g, t) for g, t in product(gold, target))


def extract_target(
    golds: list[str],
    predictions: list[str],
    formatted_doc: Doc,
    language: Language,
    gold_extraction_target: tuple[ExtractionTarget],
    pred_extraction_target: tuple[ExtractionTarget],
    aggregation_function: Callable[[list[float]], float] = max,
    extraction_mode: Literal["first_match", "extract_each_target", "first_fallback"] = "first_match",
    fallback_mode: Literal["no_fallback", "first_match", "any_match"] = "no_fallback",
    precision: int = 6,
) -> float:
    # Try each target type in order
    gold_extraction_regexes = get_extraction_regexes(formatted_doc, gold_extraction_target, language)
    pred_extraction_regexes = get_extraction_regexes(formatted_doc, pred_extraction_target, language)

    extracted_predictions = [
        extract_target_from_pred(pred, pred_extraction_regexes, extraction_mode, fallback_mode) for pred in predictions
    ]
    extracted_golds = [extract_target_from_pred(gold, gold_extraction_regexes, extraction_mode, fallback_mode) for gold in golds]

    # Assert on empty gold and warn on empty pred
    assert any(len(g) == 0 for g in extracted_golds), f"No gold targets found. Gold: {golds}, Pred: {predictions}"
    if all(len(p) == 0 for p in extracted_predictions):
        hlog_warn(f"No predictions found. Gold: {golds}, Pred: {predictions}")

    if formatted_doc.specific is None:
        formatted_doc.specific = {}

    formatted_doc.specific["extracted_predictions"] = extracted_predictions
    formatted_doc.specific["extracted_golds"] = extracted_golds

    return aggregation_function(
        (1.0 if any(compare_gold_target(gold, pred, precision) for gold in extracted_golds) else 0.0)
        for pred in extracted_predictions
    )


def multilingual_extractive_match_metric(
    language: Language,
    gold_extraction_target: tuple[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: tuple[ExtractionTarget] = (ExprExtractionConfig(),),
    aggregation_function: Callable[[list[float]], float] = max,
    extraction_mode: Literal["first_match", "extract_each_target", "first_fallback"] = "first_match",
    fallback_mode: Literal["no_fallback", "first_match", "any_match"] = "no_fallback",
    precision: int = 6,
) -> SampleLevelMetric:
    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=partial(
            extract_target,
            language=language,
            gold_extraction_target=gold_extraction_target,
            pred_extraction_target=pred_extraction_target,
            aggregation_function=aggregation_function,
            extraction_mode=extraction_mode,
            fallback_mode=fallback_mode,
            precision=precision,
        ),
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

# llm_judge_mixeval_multichoice_flow_judge = SampleLevelMetricGrouping(
#     metric_name=["llm_judge_mixeval_flow"],
#     higher_is_better={"judge_score_flow": True},
#     category=MetricCategory.LLM_AS_JUDGE,
#     use_case=MetricUseCase.SUMMARIZATION,
#     sample_level_fn=JudgeLLMMixEval(
#         judge_model_name="flowaicom/Flow-Judge-v0.1",
#         template=flow_judge_for_multichoice_template,
#         process_judge_response=process_judge_response,
#         judge_backend="vllm",
#         short_judge_name="flow",
#     ).compute,
#     corpus_level_fn={
#         "judge_score_flow": np.mean,
#     },
# )
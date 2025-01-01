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

import os
import re
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import product
from typing import Callable, Literal, Sequence

import numpy as np
import sympy
from latex2sympy2_extended import NormalizationConfig, normalize_latex
from latex2sympy2_extended import latex2sympy as parse_latex
from sympy import Basic, FiniteSet, Interval, MatrixBase, MatrixExpr, Set
from sympy.core.relational import Relational
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
)
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.utils.formulation import ChoicePrefix, get_prefix
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


class TimeoutException(Exception):
    pass


def timeout(timeout_seconds: int = 10):
    """
    A decorator that applies a timeout to the decorated function.
    On Unix: uses signal-based alarm.
    On Windows: uses a multiprocessing-based approach.

    Preferably the unix approach is better, because we don't have to spawn a new process,
    but it's not available on windows.
    """
    if os.name == "posix":
        # Unix-like approach: signal.alarm

        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutException("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    # Cancel the alarm and restore previous handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator

    else:
        # Windows approach: use multiprocessing
        from multiprocessing import Process, Queue

        def decorator(func):
            def wrapper(*args, **kwargs):
                q = Queue()

                def run_func(q, args, kwargs):
                    try:
                        result = func(*args, **kwargs)
                        q.put((True, result))
                    except Exception as e:
                        q.put((False, e))

                p = Process(target=run_func, args=(q, args, kwargs))
                p.start()
                p.join(timeout_seconds)

                if p.is_alive():
                    # Timeout: Terminate the process
                    p.terminate()
                    p.join()
                    raise TimeoutException("Operation timed out!")

                # If we got here, the process completed in time.
                success, value = q.get()
                if success:
                    return value
                else:
                    # The child raised an exception; re-raise it here
                    raise value

            return wrapper

        return decorator


# Small cache, to catche repeated calls invalid parsing
@lru_cache(maxsize=20)
@timeout(timeout_seconds=5)
def parse_latex_with_timeout(latex: str):
    return parse_latex(latex, is_real=not should_treat_as_complex(latex))


@lru_cache(maxsize=20)
@timeout(timeout_seconds=5)
def parse_expr_with_timeout(expr: str):
    return parse_expr(expr)


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


def extract_expr(match: re.Match) -> tuple[str | sympy.Expr | None, str]:
    # First combine the number
    groups = match.groupdict()
    # This musst always exist
    expr = groups["expr"]
    integer = next((val for name, val in groups.items() if name.startswith("integer") and val), None)
    decimal = next((val for name, val in groups.items() if name.startswith("decimal") and val), None)

    is_percentage = True if groups.get("percent", None) else False

    if integer:
        # Remove thousand separators and convert to float
        num_str = integer.translate(str.maketrans("", "", ", "))

        if decimal:
            # Add decimal part if present
            num_str += f"{decimal.replace(',', '.')}"
        number = sympy.Number(num_str)
        if is_percentage:
            number = number / sympy.Number(100)
        return number, expr
    elif decimal:
        # Just decimal part, convert to float, make sure to use sympy directly to avoid floating point errors
        number = sympy.Float(f"0{decimal}")
        if is_percentage:
            number = number / sympy.Number(100)
        return number, expr

    # Otherwise just return the expression
    # Remove new lines and spaces
    try:
        return parse_expr_with_timeout(expr.replace("\n", " ").replace(" ", " ").replace("^", "**")), expr
    except:
        return None, expr


@lru_cache(maxsize=1000)
def extract_latex(match: re.Match, target_type: LatexExtractionConfig) -> tuple[sympy.Expr | str | None, str]:
    latex_group, latex = next(
        ((name, val) for name, val in match.groupdict().items() if name.startswith("latex") and val), ("", "")
    )

    normalized_latex = normalize_latex(
        latex,
        NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed=True,
            equations=True,
        ),
    )

    try:
        parsed_latex = parse_latex_with_timeout(normalized_latex)
    except:
        return None, normalized_latex
    return parsed_latex, normalized_latex


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
        # We never try to match without prefixes as otherwise it will easily match partials
        regexes.append(f"({expr_prefix_re})(?P<expr>{expr_re})({expr_suffix_re})")
        regexes.append(f"({expr_prefix_re})(?P<expr>{number_re})({expr_suffix_re})")

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
    latex_boxed = r"(?P<latexBoxed>\\boxed{.+})"  # Boxed number, it's fine to be as greedy as possible as we will find the correct end afterwards
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
    fallback_mode: Literal["no_fallback", "first_match", "any_match"] = "no_fallback",
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

            if str_fallback:
                fallbacks.append(str_fallback)

            if (
                extraction_mode == "first_match"
                and extracted_predictions
                or extraction_mode == "first_fallback"
                and fallbacks
            ):
                break

        # Break early if we don't want to extract all targets
        if (
            extraction_mode == "first_match"
            and extracted_predictions
            or extraction_mode == "first_fallback"
            and fallbacks
        ):
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


def sympy_numeric_eq(a: sympy.Expr | MatrixBase, b: sympy.Expr | MatrixBase, precision: int):
    # Only do this when one of the two is a float, in other cases use symbolic equality as this could lead to false positives
    # E.g we want 1/3 == 0.333333 to work
    if isinstance(a, (MatrixBase, MatrixExpr)) and isinstance(b, (MatrixBase, MatrixExpr)):
        a = a.doit()
        b = b.doit()
        # If we have matrices and one of them is only made of floats, we can use the same logic as above
        if isinstance(a, (MatrixBase)) and isinstance(b, (MatrixBase)) and a.shape == b.shape:
            return all(sympy_numeric_eq(a_elem, b_elem, precision) for a_elem, b_elem in zip(a.flat(), b.flat()))

    else:
        # If one of them is a float, we can try to use precision
        if isinstance(a, (sympy.Float)) or isinstance(b, (sympy.Float)):
            return a.doit().round(precision) == b.doit().round(precision)
        else:
            return a.doit() == b.doit()

    return False


def sympy_symbolic_eq(a: Basic | MatrixBase, b: Basic | MatrixBase) -> bool:
    try:
        a_b_diff = sympy.simplify((a - b))
        if isinstance(a_b_diff, MatrixBase) and a_b_diff.is_zero_matrix:
            return True
        elif isinstance(a_b_diff, Basic) and a_b_diff.is_zero:
            return True
    except:
        pass

    return False


def sympy_compare_finite_set(a: FiniteSet, b: FiniteSet, precision: int) -> bool:
    try:
        if a == b or a.symmetric_difference(b).is_empty:
            return True
    except:
        pass

    # This ensures it works for {1/3} and {0.333333}
    try:
        if len(a) == len(b) and all(sympy_expr_eq(a, b, precision) for a, b in zip(a, b)):
            return True
    except:
        pass

    return False


def sympy_compare_interval(a: Interval, b: Interval, precision: int) -> bool:
    return (
        a.left_open == b.left_open
        and a.right_open == b.right_open
        and sympy_expr_eq(a.start, b.start, precision)
        and sympy_expr_eq(a.end, b.end, precision)
    )


def sympy_expr_eq(a: sympy.Expr | MatrixBase, b: sympy.Expr | MatrixBase, precision: int) -> bool:
    # Start with simple str and expr comparisson as it's the fastest
    # str comparison is better, than simple eq, because it will also handle missarangments

    # Because of float comparison, we only use doit() during the string conversion, but keep the original expr
    try:
        a_doit = a.doit()
        b_doit = b.doit()
    except:
        a_doit = a
        b_doit = b

    try:
        if str(a_doit).strip() == str(b_doit).strip():
            return True
    except:
        pass

    # Support for equations
    if isinstance(a, Relational) and isinstance(b, Relational):
        # Helper to check if expressions are equivalent when flipped
        def are_flipped_inequalities_equal(a: Relational, b: Relational) -> bool:
            return sympy_expr_eq(a.lhs - a.rhs, b.rhs - b.lhs, precision)

        # Same type of relation (e.g. both <= or both >=)
        if type(a) == type(b) and sympy_expr_eq(a.lhs - a.rhs, b.lhs - b.rhs, precision):
            return True

        # Check flipped inequalities (a <= b equals b >= a)
        if (
            isinstance(a, sympy.GreaterThan)
            and isinstance(b, sympy.LessThan)
            or isinstance(a, sympy.LessThan)
            and isinstance(b, sympy.GreaterThan)
        ) and are_flipped_inequalities_equal(a, b):
            return True

        return False

    elif isinstance(a, (Set)) or isinstance(b, (Set)):
        # This way we can also evalute {1} and 1 to be equal
        a_set = a if isinstance(a, Set) else FiniteSet(a)
        b_set = b if isinstance(b, Set) else FiniteSet(b)

        # If both are finite sets, we can compare per element
        if isinstance(a_set, FiniteSet) and isinstance(b_set, FiniteSet):
            return sympy_compare_finite_set(a_set, b_set, precision)

        elif isinstance(a_set, Interval) and isinstance(b_set, Interval):
            return sympy_compare_interval(a_set, b_set, precision)

        else:
            if a_set == b_set:
                return True
            if a_set.symmetric_difference(b_set).is_empty:
                return True
            return False

    elif isinstance(a, (Basic, MatrixBase)) and isinstance(b, (Basic, MatrixBase)):
        # For expressions,
        try:
            if sympy_numeric_eq(a, b, precision):
                return True
        except:
            pass
        # Then try symbolic equality
        try:
            if sympy_symbolic_eq(a, b):
                return True
        except:
            pass

    return False


def compare_gold_target(
    gold: list[sympy.Expr | Relational | str], target: list[sympy.Expr | Relational | str], precision: int
) -> float:
    @timeout(timeout_seconds=10)
    def compare_single_extraction(gold: str | sympy.Expr | float, target: str | sympy.Expr | float) -> float:
        # Expression case

        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(target, (Basic, MatrixBase)):
            return 1.0 if sympy_expr_eq(gold, target, precision) else 0.0

        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        else:
            return 0.0
            # raise ValueError(f"Unsupported comparison between {type(gold)} and {type(target)}")

    def compare_single_extraction_wrapper(g, t):
        try:
            return compare_single_extraction(g, t)
        except TimeoutError:
            return 0.0

    return any(compare_single_extraction_wrapper(g, t) for g, t in product(gold, target))


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
    extracted_golds = [
        extract_target_from_pred(gold, gold_extraction_regexes, "first_match", "first_match") for gold in golds
    ]

    # Assert on empty gold and warn on empty pred
    # if any(len(g) == 0 for g in extracted_golds):
    #     hlog_warn(f"No gold targets found for at least one gold. Gold: {golds}, Pred: {predictions}")

    # if all(len(p) == 0 for p in extracted_predictions):
    #     hlog_warn(f"No predictions found for all predictions. Gold: {golds}, Pred: {predictions}")

    if formatted_doc.specific is None:
        formatted_doc.specific = {}

    formatted_doc.specific["extracted_predictions"] = [str(p) for p in extracted_predictions]
    formatted_doc.specific["extracted_golds"] = [str(g) for g in extracted_golds]

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


def should_treat_as_complex(latex_str: str) -> bool:
    """
    Returns True if the latex string likely contains complex numbers, matrices, or vectors.
    """
    complex_pattern = re.compile(
        r"""
        # Complex number indicators
        \\mathbb\{C\}|        # Complex number set ℂ
        \\i\b|                # Complex i
        \bi\b|                # Standalone i
        \\text\{i\}|          # Text i
        \\mathrm\{i\}|        # Roman i
        \\imath\b|            # Alternative i notation

        # Matrix operations
        \\det|                # Determinant
        \\operatorname\{tr\}| # Trace
        \\operatorname\{rank\}| # Rank
        \\text\{rank\}|
        \\arg\{|              # Complex argument
        \\Re\{|               # Real part
        \\Im\{|               # Imaginary part
        \\operatorname\{Re\}| # Real part alternate
        \\operatorname\{Im\}| # Imaginary part alternate
        \\text\{Re\}|         # Real part text
        \\text\{Im\}          # Imaginary part text
    """,
        re.VERBOSE,
    )

    return bool(complex_pattern.search(latex_str))

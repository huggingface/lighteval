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
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import groupby
from typing import Any, Literal, Sequence

import sympy
from sympy import Basic, MatrixBase, Number
from sympy.parsing import parse_expr

from lighteval.metrics.utils.math_comparison import should_treat_as_complex
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.utils.formulation import ChoicePrefix, get_prefix
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.imports import requires_latex2sympy2_extended
from lighteval.utils.language import Language
from lighteval.utils.timeout import timeout


@requires_latex2sympy2_extended
def latex_normalization_config_default_factory():
    from latex2sympy2_extended.latex2sympy2 import NormalizationConfig

    return NormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=True,
        nits=True,
        boxed=True,
        equations=True,
    )


@dataclass(frozen=True)
class LatexExtractionConfig:
    """Config for extracting latex from the prediction.

    Attributes:
        try_extract_without_anchor (bool): Whether to try extracting latex without requiring specific anchors like "answer:" or "final answer is"
        boxed_match_priority (int): Priority of the boxed match regex (-1 never, 0 first, 55 after final answer: anchor, etc...)
        normalization_config (latex2sympy2_extended.latex2sympy2.NormalizationConfig): Normalization config to use for latex extraction
    """

    try_extract_without_anchor: bool = True
    boxed_match_priority: int = 55
    normalization_config: Any = field(default_factory=latex_normalization_config_default_factory)


@dataclass(frozen=True)
class ExprExtractionConfig:
    """Config for extracting mathematical expressions.

    Attributes:
        try_extract_without_anchor (bool): Whether to try extracting expressions without requiring specific anchors like "answer:" or "final answer is"
    """

    try_extract_without_anchor: bool = True


@dataclass(frozen=True)
class IndicesExtractionConfig:
    """Config for extracting choice indices.

    Attributes:
        prefix_for_extraction (ChoicePrefix): The style to use for extracting choice indices (e.g. A,B,C or 1,2,3)
        try_extract_without_anchor (bool): Whether to try extracting indices without requiring specific anchors like "answer:" or "final answer is"
    """

    prefix_for_extraction: ChoicePrefix
    try_extract_without_anchor: bool = True


ExtractionTarget = LatexExtractionConfig | ExprExtractionConfig | IndicesExtractionConfig


# All of the regexes are cached, to avoid repeated compiling during processing of same task
@lru_cache(maxsize=1)
def lazy_expr_regex(expr_config: ExprExtractionConfig, language: Language) -> list[tuple[re.Pattern[str], int]]:
    translation_literal = TRANSLATION_LITERALS[language]

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
        r")(?P<percent>\s*(?:%|[Pp]ercent|\s*[Pp]ercentage|\s*[Pp]ct))?"
    )

    # Expressions such as 1/2
    operators = [r"\+", r"\-", r"\*", r"\×", r"\/", r"\^", r"\(", r"\)", r"\÷"]
    operators_re = "".join(operators)
    all_expr_chars = r"[\d\.\s" + operators_re + r"]"
    # Expression should have at minimum at least one operator and must start with a digit
    expr_re = rf"(?P<expr>-?\(?-?\d{all_expr_chars}*[{operators_re}]{all_expr_chars}+\)?)"

    # Punctuation regexes
    full_stop_re = rf"[{re.escape(translation_literal.full_stop)}\.]"
    comma_re = rf"[{re.escape(translation_literal.comma)}\,]"
    colon_re = rf"[{re.escape(translation_literal.colon)}\:]"
    space_re = rf"(?:\s|{re.escape(translation_literal.sentence_space)})"

    currency_units = re.escape("$€£¥₹₽₪₩₫฿₡₢₣₤₥₦₧₨₩₪₫₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾₿")
    expr_prefix_re = rf"(?:^|{space_re}|\=)(?:\*\*)?"
    expr_suffix_re = rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|\)|\$|$)"
    # Expressions must be prefixed and suffixed while, digits don't need suffix and can have currency units preceeded, this is to ensure
    # That we can extract stuff like $100 or 100m2, while we don't extract XDY2K as 2
    expr_with_anchors = rf"(?:{expr_prefix_re}{expr_re}{expr_suffix_re})"
    number_with_anchors = rf"(?:{expr_prefix_re}[{currency_units}]?{number_re})"
    expr_or_number = rf"(?:{expr_with_anchors}|{number_with_anchors})"
    regexes: list[tuple[str, int]] = []

    # Ideally we would have translation of such concept in each language
    if language == Language.ENGLISH:
        final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{expr_or_number}\.?\s?I hope"
        final_answer_prefixed_just_is = rf"(?i:final answer.{{0,100}}?)\s+is\:?{expr_or_number}"
        regexes.append((final_answer_prefixed_re, 0))
        regexes.append((final_answer_prefixed_just_is, 50))

    answer_prefix_re = rf"(?i:{translation_literal.answer})"

    # Match after the last equals with answer word - require the number pattern,
    equals_re_colon = rf"{answer_prefix_re}{colon_re}(?:.{{0,100}}=\s*|.{{0,50}}?){expr_or_number}(?!\s*=)"
    equals_re = rf"{answer_prefix_re}(?:.{{0,100}}=\s*|.{{0,50}}?){expr_or_number}(?!\s*=)"
    regexes.extend([(equals_re_colon, 100), (equals_re, 200)])

    if expr_config.try_extract_without_anchor:
        # If everything fails, try to match plain expr/number
        regexes.append((expr_with_anchors, 300))
        regexes.append((number_with_anchors, 300))

    return [(re.compile(pattern), priority) for pattern, priority in regexes]


@lru_cache(maxsize=1)
def lazy_latex_regex(latex_config: LatexExtractionConfig, language: Language) -> list[tuple[re.Pattern[str], int]]:
    # Only LaTeX expressions between delimiters
    percent_re_group = r"(?P<percent>\s*(?:\\?%|[Pp]ercent|[Pp]ercentage|[Pp]ct))"
    latex_envs_re = (
        r"("
        r"(?<!\\)\$\$(?P<latexDisplayDollar>[\s\S]+?)(?<!\\)\$\$|"  # $$...$$ (display math, can be multiline)
        r"(?<!\\)\\\[(?P<latexDisplayBracket>[\s\S]+?)(?<!\\)\\\]|"  # \[...\] (display math, can be multiline)
        r"(?<!\\|\d)\$(?P<latexInlineDollar>(?:\\[$]|[^\n$])+?)(?<!\\)\$|"  # $...$ (inline math, single line, allows escaped $), we make sure it's not preceded by a digit to minimize false positives containing dollar as a unit
        r"(?<!\\)\\\((?P<latexInlineParenthesis>[^\n]+?)(?<!\\)\\\)|"  # \(...\) (inline math, single line)
        r"(?<!\\)\[(?P<latexInlineBracket>[^\n$]+?)(?<!\\)\]"  # [....] While this is not a valid display, math LLMs like to generate it. We allow it
        rf"){percent_re_group}?"
    )

    # Match latex without environments
    latex_boxed = rf"(?P<latexBoxed>\\boxed{{.+}})\$?{percent_re_group}?"  # Boxed number, it's fine to be as greedy as possible as we will find the correct end afterwards
    simple_number = r"-?\d+(?:[.,]\d+)?"
    latex_fraction = rf"(?P<latexFraction>-?\\frac{{{simple_number}}}{{{simple_number}}})\$?{percent_re_group}?"

    translation_literal = TRANSLATION_LITERALS[language]
    colon_re = rf"[{re.escape(translation_literal.colon)}\:]"

    answer_prefix_re = rf"(?i:{translation_literal.answer})"

    # We first match boxed env, for some reason that's the most common case of output
    # Then we match the latex with environments, then we try to match the fraction
    regexes: list[tuple[str, int]] = []
    for latex_re in [latex_envs_re, latex_fraction]:
        if language == Language.ENGLISH:
            final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{latex_re}\.?\s?I hope"
            final_answer_prefixed_just_is = rf"(?i:final answer.{{0,100}}?)\s+is\:?\s*{latex_re}"
            regexes.append((final_answer_prefixed_re, 0))
            regexes.append((final_answer_prefixed_just_is, 50))

        # Match with answer word - higher priority than plain latex
        answer_re_colon = f"{answer_prefix_re}{colon_re}.{{0,50}}?{latex_re}"
        answer_re = f"{answer_prefix_re}.{{0,50}}?{latex_re}"

        regexes.extend([(answer_re_colon, 100), (answer_re, 200)])

        # Match plain LaTeX - lowest priority
        if latex_config.try_extract_without_anchor:
            regexes.append((latex_re, 300))

    if latex_config.boxed_match_priority >= 0:
        regexes.append((latex_boxed, latex_config.boxed_match_priority))

    return [(re.compile(pattern, re.DOTALL), priority) for pattern, priority in regexes]


@lru_cache(maxsize=10)
def lazy_indices_regex(
    indices_config: IndicesExtractionConfig, len_choices: int, language: Language
) -> list[tuple[re.Pattern[str], int]]:
    translation_literal = TRANSLATION_LITERALS[language]
    # First get indices to predict
    indices = get_prefix(indices_config.prefix_for_extraction, translation_literal)[:len_choices]
    indice_str_re = f"(?P<indices>{'|'.join([re.escape(i) for i in indices])})"

    # The answer keys are either surrounded with <space>**answer**., or '<space>answer.' or the same without the dot
    full_stop_re = rf"[{re.escape(translation_literal.full_stop)}\.]"
    comma_re = rf"[{re.escape(translation_literal.comma)}\,]"
    colon_re = rf"[{re.escape(translation_literal.colon)}\:]"
    space_re = re.escape(translation_literal.sentence_space)

    answer_prefix_re = rf"(^|{space_re})(?:\*\*)?"
    answer_suffix_re = rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|$)"
    answer_re = f"{answer_prefix_re}{indice_str_re}{answer_suffix_re}"
    answer_re_start = rf"^(?:\*\*)?{indice_str_re}{answer_suffix_re}"

    answer_word = f"(?i:{translation_literal.answer})"

    regexes = []

    if language == Language.ENGLISH:
        final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{indice_str_re}\.?\s?I hope"
        # To allow stuff like "final answer is to your question"
        final_answer_prefixed_just_is = rf"(?i:final answer.{{0,100}}?)\s+is\:?\s*{indice_str_re}"
        regexes.extend(
            [
                (final_answer_prefixed_re, 0),
                (final_answer_prefixed_just_is, 50),
            ]
        )

    regexes.extend(
        [
            # Most specific patterns first
            (f"{answer_word}{colon_re}.{{0,50}}?{answer_re}", 100),
            # Answer word patterns
            (f"{answer_word}.{{0,50}}?{answer_re}", 150),
            # Start of line patterns
            (answer_re_start, 200),
        ]
    )

    if indices_config.try_extract_without_anchor:
        # Less specific patterns
        regexes.extend(
            [
                (answer_re, 250),
                (indice_str_re, 300),
            ]
        )

    return [(re.compile(pattern), priority) for pattern, priority in regexes]


def get_extraction_regexes(
    formatted_doc: Doc, target_types: Sequence[ExtractionTarget], language: Language
) -> list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]]:
    extraction_regexes: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]] = [
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


# Small cache, to catche repeated calls invalid parsing
@lru_cache(maxsize=20)
@timeout(timeout_seconds=5)
@requires_latex2sympy2_extended
def parse_latex_with_timeout(latex: str):
    from latex2sympy2_extended.latex2sympy2 import latex2sympy

    return latex2sympy(latex, is_real=not should_treat_as_complex(latex), convert_degrees=False)


@lru_cache(maxsize=20)
@timeout(timeout_seconds=5)
def parse_expr_with_timeout(expr: str):
    return parse_expr(expr, evaluate=False)


def extract_expr(match: re.Match) -> tuple[str | sympy.Expr | None, str]:
    # First combine the number
    groups = match.groupdict()
    # Expr group will always exist because every regex has it
    expr = groups.get("expr", "")
    integer = next((val for name, val in groups.items() if name.startswith("integer") and val), "")
    decimal = next((val for name, val in groups.items() if name.startswith("decimal") and val), "")

    is_percentage = True if groups.get("percent", None) else False

    if integer or decimal:
        # This makes sure we can convert numbers like 0001 to 1. Do note that this can convert 0 to '', so we assume an empty string was 0 and convert it back afterwards.
        integer = integer.translate(str.maketrans("", "", ", ")).lstrip("0")
        if len(integer) == 0:
            integer = "0"

        decimal = decimal.replace(",", ".")
        number_str = f"{integer}{decimal}"
        number = Number(number_str)

        if is_percentage:
            number = convert_to_pct(number)
        return number, number_str

    # Otherwise just return the expression
    # Remove new lines and spaces
    if expr:
        try:
            return parse_expr_with_timeout(expr.replace("\n", " ").replace("^", "**")), expr
        except:  # noqa: E722
            pass
    return None, expr


def convert_to_pct(number: Number):
    return sympy.Mul(number, sympy.Rational(1, 100), evaluate=False)


@lru_cache(maxsize=1000)
@timeout(timeout_seconds=5)
@requires_latex2sympy2_extended
def extract_latex(match: re.Match) -> tuple[sympy.Expr | str | None, str]:
    from latex2sympy2_extended.latex2sympy2 import NormalizationConfig, normalize_latex

    latex = next((val for name, val in match.groupdict().items() if name.startswith("latex") and val), "")
    is_percentage = True if match.group("percent") else False

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
        if is_percentage:
            parsed_latex = convert_to_pct(parsed_latex)
    except:  # noqa: E722
        return None, normalized_latex
    return parsed_latex, normalized_latex


def extract_match(match: re.Match, target_type: ExtractionTarget) -> tuple[Basic | MatrixBase | str | None, str]:
    """Extracts the match from the regex match.

    Args:
        match (re.Match): The regex match object containing the extracted text
        target_type (ExtractionTarget): The type of extraction to perform (latex, expression, or indices)

    Returns:
        tuple[Basic | MatrixBase | str | None, str]: A tuple containing:
            - The extracted and parsed value (if successful) or None (if parsing failed)
            - The string representation of the extracted text
    """
    if isinstance(target_type, LatexExtractionConfig):
        return extract_latex(match)
    elif isinstance(target_type, ExprExtractionConfig):
        return extract_expr(match)
    elif isinstance(target_type, IndicesExtractionConfig):
        return match.group("indices"), match.group("indices")


def extract_target_from_pred(
    pred: str,
    target_res: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
    fallback_mode: Literal["no_fallback", "first_match"] = "no_fallback",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
):
    """Extracts targets from a prediction string using regex patterns.
    Returns first sucesffuly extracted match.

    Args:
        pred (str): The prediction string to extract from
        target_res (list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]]): List of regex patterns and their priorities for each target type
        fallback_mode (Literal["no_fallback", "first_match"], optional): How to handle extraction failures. Defaults to "no_fallback".
            - "no_fallback": Return only successfully parsed match
            - "first_match": Additionaly Include the first string match no matter how parsing finished
        extraction_mode (Literal["first_match", "any_match"], optional): How to handle extraction failures. Defaults to "any_match".
            - "first_match": Only tries to extract the first match
            - "any_match": Tries to extract any match

    Returns:
        list: List of extracted predictions, with first fallbac string appended if fallback_mode is "first_match"
    """
    extracted_predictions = []
    fallbacks = []

    # Get all patterns and sort by priority
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in target_res
        for pattern, priority in target_patterns
    ]
    match_found = False

    # Group patterns by priority using itertools.groupby
    for _, patterns_group in groupby(sorted(all_patterns, key=lambda x: x[2]), key=lambda x: x[2]):
        # Find all matches for each pattern in this priority group
        matches_with_pos = (
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred)
        )

        # Sort matches by end position (rightmost first) and then by start position (leftmost first)
        matches_with_pos = sorted(matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True)

        # Try to extract from each match, starting from rightmost
        for match, _, _, target_type in matches_with_pos:
            extracted_match, str_fallback = extract_match(match, target_type)
            match_found = True

            if str_fallback:
                fallbacks.append(str_fallback)

            if extracted_match is not None:
                extracted_predictions.append(extracted_match)
                break

            if extraction_mode == "first_match":
                break

        # If we found something and we're in first_match mode, stop processing other priorities
        if extracted_predictions or (match_found and extraction_mode == "first_match"):
            break

    if fallback_mode == "first_match" and fallbacks:
        extracted_predictions += [fallbacks[0]]

    return extracted_predictions

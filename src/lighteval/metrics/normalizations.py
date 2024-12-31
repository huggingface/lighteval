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
import string
import sys
import unicodedata
from dataclasses import dataclass
from typing import Callable

from lighteval.metrics.utils.linguistic_tokenizers import get_word_tokenizer
from lighteval.utils.language import Language


def remove_outer_braces(text: str) -> str:
    pairs = ["{}", "[]", "()"]
    for l, r in pairs:
        if text.startswith(l) and text.endswith(r):
            return text[len(l) : -len(r)]
    return text


# From HELM
def helm_normalizer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def homogeneize_numbers(text: str) -> str:
        """Casts text to float to test if it's a number, then casts back to string.
        This allows equal numbers formatted differently (1.0 vs 1 for ex) to be considered
        equal. This comes from Harness DROP - check if it causes a discrep in QuAC
        """
        try:
            return str(float(text))
        except ValueError:
            return text

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    def _tokenize(text):
        return re.split(" |-", text)

    tokens = [white_space_fix(remove_articles(homogeneize_numbers(remove_punc(lower(t))))) for t in _tokenize(text)]
    return " ".join([t for t in tokens if t != ""]).strip()


def harness_triviaqa_normalizer(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def bigbench_normalizer(text: str):
    return text.replace(" . ", ".\n")


def remove_braces(text: str) -> str:
    if text.startswith("{"):
        text = text[1:]
    if text.endswith("}"):
        text = text[:-1]
    return text


def remove_braces_and_strip(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        text = text[1:]
    if text.endswith("}"):
        text = text[:-1]
    return text


units = [
    "integer" "point",
    "feet",
    "sue",
    "digit",
    "pound",
    "meal",
    "edge",
    "student",
    "children ticket",
    "multiple",
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m square",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "cent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

# We sort here to that when matching from right the longest units are matched first
# E.g "percent" is matched before "cent"

units_regex = re.compile("|".join([f"(?=\\s)(?:{unit}(?:s|es)?)($|\\W)" for unit in units]))

to_remove_regex = re.compile(
    r"\\mathrm\{th\}|"
    r"\{,\}|"
    r"(?<!\\)\\\s|"  # backslash with whitespace (but not in matrix line breaks)
    r"\\\$|\$|"  # dollar signs
    r",\\!|"  # comma with inverse space
    r"(?<=\s)(and)(?=\s)|"  # "and" with whitespace
    r"(?<!\\)[\"\']"
)

to_replace_patterns = [
    # (name, pattern, replacement)
    # Not really needed only for units
    ("math", r"\\math(?:rm|it|bf)", r"\text"),
    ("text", r"\\text(?:normal|bf|it|rm)", r"\text"),
    ("frac", r"\\(?:d|t|c)frac", r"\frac"),
    ("decimal_space", r"\s\.", r" 0."),
    ("decimal_brace", r"\{\.", r"{0."),
    ("approx", r"\~\=", r"\approx"),
    ("infinity", r"infinity", r"\infty"),
    ("inf", r"((?<!\\)inf(?!inity))", r"\infty"),
    ("sqrt", r" sqrt", r"\sqrt"),
]


# Create regex with named groups
pattern = "|".join(f"(?P<{name}>{pattern})" for name, pattern, _ in to_replace_patterns)
to_replace_regex = re.compile(pattern)

# Create lookup dictionary for replacements
replacements = {name: replacement for name, _, replacement in to_replace_patterns}


def replace(match):
    # Find which group matched
    # Get corresponding replacement from dict
    return replacements[match.lastgroup]


def replace_in_latex(text: str) -> str:
    return to_replace_regex.sub(replace, text)


# Make sure we don't break line breaks


command_slash_fix_regex = re.compile(r"\\\\(?=[a-zA-Z])")


def extract_last_boxed_content(text: str) -> str:
    """
    Find and extract the content of the last \\boxed{...} or \\fbox{...} element from a string.

    Example:
    >>> extract_last_boxed_content("Some text \\boxed{\\frac{2}{3}}")
    "\\frac{2}{3}"
    >>> extract_last_boxed_content("\\boxed 123")
    "123"
    >>> extract_last_boxed_content("No box here")
    ""
    """

    # Then look for \\boxed{...} or \\fbox{...}
    env = "\\boxed"
    left_idx = text.rfind(env)
    if left_idx < 0:
        env = "\\fbox"
        left_idx = text.rfind(env)
        if left_idx < 0:
            return text
    left_idx += len(env)

    # If the next character is a brace remove it, otherwise it's a \\boxed {content}
    if len(text) > left_idx and text[left_idx] != "{":
        # If there is no opening brace, it's a \\boxed {content}
        return text[left_idx:].lstrip()

    # Find matching closing brace
    i = left_idx
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                # Extract content between braces (+1 to remove the opening brace)
                return text[left_idx + 1 : i]
        i += 1

    # Otherwise, it's no a valid latex
    return text


def math_normalizer(text: str, skip_unit: bool = False) -> str:  # noqa C901
    """Source: https://github.com/hendrycks/math"""

    def _fix_fracs(text: str) -> str:
        """
        Fix the formatting of fractions in the given text.
        Copied from: https://github.com/hendrycks/math/blob/357963a7f5501a6c1708cf3f3fb0cdf525642761/modeling/math_equivalence.py#L1

        Args:
            text (str): The input text.

        Returns:
            str: The text with properly formatted fractions.

        Examples:
            >>> _fix_fracs("\\frac12")
            "\\frac{1}{2}"
            >>> _fix_fracs("\\frac{3}{4}")
            "\\frac{3}{4}"
            >>> _fix_fracs("\\frac1{2}")
            "\\frac{1}{2}"
        """
        substrs = text.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            for substr in substrs[1:]:
                # This allows use to have \\frac{1}{2} and \\ frac1{2}
                substr = substr.lstrip()
                new_str += "\\frac"
                if len(substr) > 0 and substr[0] == "{":
                    new_str += substr

                elif len(substr) < 2:
                    return text
                else:
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        text = new_str
        return text

    def _fix_a_slash_b(text: str) -> str:
        """Source: https://github.com/hendrycks/math
        Reformat fractions formatted as a/b to \\frac{a}{b}.
        Example:
        >>> _fix_a_slash_b("2/3")
        \frac{2}{3}
        """
        if len(text.split("/")) != 2:
            return text
        a_str = text.split("/")[0]
        b_str = text.split("/")[1]
        try:
            a = int(a_str)
            b = int(b_str)
            assert text == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return text

    def _fix_sqrt(text: str) -> str:
        """Source: https://github.com/hendrycks/math
        Reformat square roots.
        Example:
        >>> _fix_sqrt("\\sqrt3")
        \\sqrt{3}
        """
        if "\\sqrt" not in text:
            return text
        splits = text.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            split = split.lstrip()
            if len(split) > 0 and split[0] not in ["{", "["]:
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def _remove_text_formatting(text: str) -> str:
        """Remove text formatting commands like \text{}, \textbf{}, \\overline{}, and \boxed{}.
        Also ensures math expressions are properly wrapped in single $ signs.

        Args:
            text (str): The text to process

        Returns:
            str: Text with formatting commands removed and math properly delimited

        Examples:
            - Input: 'outer $\\text{inner}$ text'
            Output: 'outer $inner$ text'
            - Input: '$\\textbf{bold math}$'
            Output: '$bold math$'

            - Input: '$\\overline{x + y}$'
            Output: '$x + y$'
        """
        text = re.sub(r"(\\text\{)(.*?)(\})", "\\2", text)  # remove \text{...}
        text = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", text)  # remove \overline{...}
        return text

    def _fix_malformed_operators(text: str) -> str:
        """
        Fix malformed operators in the given text.
        """
        expr_str = text
        # Usage of () instead of {}
        expr_str = re.sub(r"\^\s?\((.*?)\)", r"^{\1}", expr_str)
        expr_str = re.sub(r"sqrt\s?\((.*?)\)", r"\\sqrt{\1}", expr_str)
        expr_str = re.sub(r"\\frac\s?(\d)\s?(\d+)", r"\\frac{\1}{\2}", expr_str)
        expr_str = re.sub(r"\\log_\s?(\d)\s?(\d+)", r"\\log_{\1}{\2}", expr_str)
        expr_str = re.sub(r"\\frac\s?{(.*?)}\s?(\d)", r"\\frac{\1}{\2}", expr_str)
        expr_str = re.sub(r"\\frac\s?(\d)\s?{(.*?)}", r"\\frac{\1}{\2}", expr_str)
        expr_str = re.sub(r"\\sqrt\s?(\d)", r"\\sqrt{\1}", expr_str)
        expr_str = expr_str.replace(" sqrt", "\\sqrt")
        return expr_str

    # First extract the last boxed content
    text = extract_last_boxed_content(text)

    # Take last expr after the =, it's important to do this after finding the boxed env as otherwise we might get the wrong expr
    # Remove new lines and simplify tabs
    text = text.replace("\n", " ").replace("\t", " ")

    # Sometimes the \\ are doubled so we substitute them, the only case where they they should be doubled is when they are line breaks.
    # In such case we leave them as is. Therefore we only apply this in case of non matrix and only when the slash is followed by a letter (to catch commands)
    if "matrix" not in text:
        text = command_slash_fix_regex.sub(r"\\", text)

    # Remoove useless latex commands
    text = to_remove_regex.sub("", text)

    text = replace_in_latex(text)

    # Split on =, by now all <=/>=/!= are replaced with \le/\ge/\ne
    # This means that we can never detect an equation but it's what it's
    text = re.split(r"(?<!\\|\<|\!|\>)=", text)[-1]

    # Split on approximate, we want to take everything but last part because we don't care about approximate
    approx_split = re.split(r"\\approx", text)
    # We take the second last part because we don't care about approximate
    if len(approx_split) > 1:
        text = approx_split[-2]

    # Remove the units and possibly the superscript (for things like m^2)
    _text = re.sub(r"\\text{.*?}(\^\d|\{\^\d\})?$", "", text).strip()
    if _text != "" and _text != text:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        text = _text

    if not skip_unit:
        # Remove unit: texts, we do thiss twice too remove stuff like meter square
        for _ in range(2):
            _text = units_regex.sub(r"\1\2", text)
        if _text != "" and _text != text:
            text = _text

    if len(text) > 0 and text[0] == ".":
        text = "0" + text

    # Fix malformed operators
    text = _fix_malformed_operators(text)

    # fix sqrt3 --> sqrt{3}
    text = _fix_sqrt(text)

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    text = _fix_fracs(text)

    # manually change 0.5 --> \frac{1}{2}
    if text == "0.5":
        text = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    text = _fix_a_slash_b(text)

    return text.strip()


def gsm8k_normalizer(text: str) -> str:
    """
    from https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28

    Args:
        text (str): input text

    Returns:
        str: Output text, either the number found in the text or "[invalid]" if
        no number was found
    """
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    match = ANS_RE.search(text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")}.union(
    string.punctuation
)

_ARTICLE_PATTERNS = {
    Language.ENGLISH: r"\b(a|an|the)\b",
    Language.SPANISH: r"\b(el|la|los|las|un|una|unos|unas)\b",
    Language.PORTUGUESE: r"\b(o|a|os|as|um|uma|uns|umas)\b",
    Language.ITALIAN: r"\b(il|lo|la|i|gli|le|un|uno|una)\b",
    Language.FRENCH: r"\b(le|la|les|l'|un|une|des)\b",
    Language.GERMAN: r"\b(der|die|das|den|dem|des|ein|eine|einer|eines|einem|einen)\b",
    Language.FINNISH: r"\b(se|yksi|yks)\b",
    Language.GREEK: r"\b(ὁ|οἱ|τοῦ|τῶν|τόν|τούς|ὦ|ἡ|αἱ|τῆς|τῶν|τήν|τάς|τό|τά|τοῦ|τῶν|τό|τά)\b",
    Language.NORWEGIAN: r"\b(en|ei|et|den|det|de)\b",
    Language.SWEDISH: r"\b(en|ett|den|det|de)\b",
    Language.TURKISH: r"\b(bir)\b",
    Language.DUTCH: r"\b(de|het|een)\b",
    Language.HUNGARIAN: r"\b(a|az|egy)\b",
    Language.CATALAN: r"\b(el|la|els|les|un|una|uns|unes)\b",
    Language.HEBREW: r"\b(ה)\b",
    Language.GALICIAN: r"\b(o|a|os|as|un|unha|uns|unhas)\b",
}


def remove_articles(text: str, lang: Language) -> str:
    """
    Removes definite and indefinite articles from the text.
    Generated using LLM then manually checked by non-expert.
    We currently only support languages that don't blend articles.
    If you are a native speaker of a language where articles are blended,
    we would appreciate your contribution!
    """
    pattern = _ARTICLE_PATTERNS.get(lang)
    return re.sub(pattern, " ", text) if pattern else text


def remove_punc(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCT)


def get_multilingual_normalizer(lang: Language, lower: bool = True) -> Callable[[str], str]:
    tokenizer = get_word_tokenizer(lang)

    def _inner_normalizer(text: str) -> str:
        text = remove_articles(text, lang)
        text = remove_punc(text)
        if lower:
            text = text.lower()

        tokens = tokenizer.word_tokenize(text)
        return " ".join(tokens)

    return _inner_normalizer


# Loglikelihood normalization
@dataclass
class LogProbPMINorm:
    """
    Performs Pointwise mutual information normalization. log_likelihood_conditioned - log_likelihood_unconditioned.
    Useful when answer contains generally unlikely tokens.
    """

    name: str = "norm_pmi"

    pass


@dataclass
class LogProbTokenNorm:
    """
    Performs token level normalization. log_likelihood/token_length.
    Useful for non-english languages.
    """

    name: str = "norm_token"
    pass


@dataclass
class LogProbCharNorm:
    """
    Performs character level normalization. log_likelihood/char_length
    ignore_first_space (bool, optional): Whether to ignore the first token's log prob (if it's a space only). Defaults to False.
        The only case when it should be True is when the possible choices (for example `A`,`B` ...) have an extra
        space added in front of them to manage tokenization issues (` A`, ` B`, ...) for some models.
    """

    name: str = "norm"

    ignore_first_space: bool = False


LogProbNormalization = LogProbCharNorm | LogProbTokenNorm | LogProbPMINorm


def normalize_log_probs(
    normalization: LogProbNormalization,
    choices_logprob: list[float],
    unconditioned_logprob: list[float] | None,
    choices_text: list[str] | None,
    choices_tokens: list[list[int]] | None,
) -> list[float]:
    normalized_log_probs = choices_logprob
    match normalization:
        case LogProbCharNorm(ignore_first_space=True):
            assert choices_text is not None, "choices_text must be provided for character normalization"
            normalized_log_probs = [
                choices_logprob[ix] / (len(choice) - 1 if choice[0] == " " else len(choice))
                for ix, choice in enumerate(choices_text)
            ]
        case LogProbCharNorm(ignore_first_space=False):
            assert choices_text is not None, "choices_text must be provided for character normalization"
            normalized_log_probs = [choices_logprob[ix] / len(choice) for ix, choice in enumerate(choices_text)]
        case LogProbTokenNorm():
            assert choices_tokens is not None, "choices_tokens must be provided for token normalization"
            normalized_log_probs = [
                choices_logprob[ix] / len(choices_tokens[ix]) for ix in range(len(choices_logprob))
            ]
        case LogProbPMINorm():
            assert unconditioned_logprob is not None, "unconditioned_logprob must be provided for PMI normalization"
            normalized_log_probs = [
                choices_logprob[ix] - unconditioned_logprob[ix] for ix in range(len(choices_logprob))
            ]

    return normalized_log_probs

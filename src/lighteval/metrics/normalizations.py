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
from dataclasses import dataclass


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


def math_normalizer(text: str) -> str:  # noqa C901
    """Source: https://github.com/hendrycks/math"""

    def _remove_boxed(text: str | None) -> str:
        """
        Extract the text within a \\boxed{...} environment.
        Example:
        >>> _remove_boxed(\\boxed{\\frac{2}{3}})
        \\frac{2}{3}
        """
        if text is None:
            return ""
        if "\\boxed " in text:
            left = "\\boxed "
            assert text[: len(left)] == left
            return text[len(left) :]

        left = "\\boxed{"

        assert text[: len(left)] == left
        assert text[-1] == "}"

        return text[len(left) : -1]

    def _last_boxed_only_string(text: str) -> str | None:
        """Extract the last \\boxed{...} or \\fbox{...} element from a string."""
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            if text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = text[idx : right_brace_idx + 1]

        return retval

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
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return text
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

    def _remove_right_units(text: str) -> str:
        """
        Removes unit descriptions from LaTeX-formatted text, where units are indicated by "\\text{ }".
        This function splits the text at each "\\text{ " and returns the part before the first occurrence,
        effectively discarding any units and additional text following this pattern. This function also
        trims any trailing whitespace left after removing units.

        Args:
            text (str): The input string potentially containing LaTeX-style unit descriptions.

        Returns:
            str: The text with unit descriptions removed.

        Examples:
            - Input: '50.5 \\text{ kg}'
            Output: '50.5'

            - Input: 'The mass is 20 grams'
            Output: 'The mass is 20 grams'

            - Input: 'The object weighs 30.2 \\text{ lbs} and is 15 \\text{ inches} long'
            Output: 'The object weighs 30.2'

            - Input: '\\text{ unit without preceding text}'
            Output: ''

        Note:
            This function assumes that "\\text{ " is only used to denote units. It will remove all text
            following the first occurrence of "\\text{ ", including any further text and units that might
            appear in complex sentences.
        """
        # Check for "\\text{ " and split the text at each occurrence
        if "\\text{ " in text:
            splits = text.split("\\text{ ")
            # Return only the first part which is assumed to contain the main content without units
            return splits[0].rstrip()
        else:
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
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    text = _remove_boxed(_last_boxed_only_string(text))

    to_replace_1 = [
        ("\n", ""),  # linebreaks
        ("\\!", ""),  # remove inverse spaces
        ("\\\\", "\\"),  # replace \\ with \
        ("tfrac", "frac"),  # replace tfrac and dfrac with frac
        ("dfrac", "frac"),
        ("\\left", ""),  # remove \left and \right
        ("\\right", ""),
        ("^{\\circ}", ""),  # Remove circ (degrees)
        ("^\\circ", ""),
        ("\\$", ""),  # remove dollar signs
    ]

    for input_str, output_str in to_replace_1:
        text = text.replace(input_str, output_str)

    # remove units (on the right)
    text = _remove_right_units(text)

    to_replace_2 = [
        ("\\%", ""),  # remove percentage
        (r"\%", ""),
        (
            " .",
            " 0.",
        ),  # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the text
        ("{.", "{0."),
    ]
    for input_str, output_str in to_replace_2:
        text = text.replace(input_str, output_str)

    # if empty, return empty text
    if len(text) == 0:
        return text
    if text[0] == ".":
        text = "0" + text

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(text.split("=")) == 2:
        if len(text.split("=")[0]) <= 2:
            text = text.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    text = _fix_sqrt(text)

    # remove spaces
    text = text.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    text = _fix_fracs(text)

    # manually change 0.5 --> \frac{1}{2}
    if text == "0.5":
        text = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    text = _fix_a_slash_b(text)

    return text


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

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


def math_normalizer(text: str, is_gold: bool = False) -> str:  # noqa C901
    """Source: https://github.com/hendrycks/math"""

    def _remove_boxed(text: str) -> str:
        """
        Extract the text within a \\boxed{...} environment.
        Example:
        >>> _remove_boxed(\\boxed{\\frac{2}{3}})
        \\frac{2}{3}
        """
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
                    a, b = substr
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
        """Source: https://github.com/hendrycks/math
        Remove units (on the right).
        "\\text{ " only ever occurs (at least in the val set) when describing units.
        """
        if "\\text{ " in text:
            splits = text.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
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

    if is_gold:
        text = _remove_boxed(_last_boxed_only_string(text))
    else:
        indices = [pos for pos, char in enumerate(text) if char == "$"]
        if len(indices) > 1:
            text = text[indices[0] + 1 : indices[-1]]

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


def math_normalizer_gold(text: str) -> str:
    return math_normalizer(text, True)


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

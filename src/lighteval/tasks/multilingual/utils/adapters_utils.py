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

from functools import partial, reduce
from typing import Literal

from lighteval.tasks.templates.utils.formatting_utils import PUNCT
from lighteval.tasks.templates.utils.translation_literals import TranslationLiterals


MULTICHOICE_JOIN_VARIANT = Literal["COMMA", "NEW_LINE"]


def multichoice_join(choices: list[str], variant: MULTICHOICE_JOIN_VARIANT, translation_literals: TranslationLiterals):
    """
    Joins the choices with the appropriate separator.
    """
    separator: str
    if variant == "COMMA":
        separator = f"{translation_literals.comma}{translation_literals.word_space}"
    elif variant == "NEW_LINE":
        separator = "\n"

    return separator.join(choices)


def multichoice_to_single_choice(
    choices: list[str],
    gold_idx: list[int],
    join_variant: MULTICHOICE_JOIN_VARIANT,
    translation_literals: TranslationLiterals,
):
    """
    Converts from multi-choice format to single-choice format, by joining the correct choices with the appropriate separator.
    Args:
        choices (list[str]): List of choices.
        gold_idx (list[int]): List of indices of the correct choices.
        join_variant (MULTICHOICE_JOIN_VARIANT): Variant of the separator to join the choices.
        translation_literals (TranslationLiterals): Translation literals.
    Returns:
        tuple[list[str], list[int]]: Tuple of the new choices and the new gold index.
    """
    if len(gold_idx) == 1:
        return choices, gold_idx

    multichoice_joiner = partial(multichoice_join, variant=join_variant, translation_literals=translation_literals)

    import random

    correct_choice = multichoice_joiner([choices[i] for i in gold_idx])
    incorrect_choices = [choice for i, choice in enumerate(choices) if i not in gold_idx]

    new_choices = incorrect_choices + [correct_choice]
    random.shuffle(new_choices)

    new_gold_idx = new_choices.index(correct_choice)

    return new_choices, [new_gold_idx]


def extract_answers_from_string(answer_string: str, answer_prefixes: list[str]) -> tuple[int, dict[str, str]] | None:
    """
    Attempts to extract answers from the answer_string. The answers are identified by being prefixed with answer prefixes.
    The extraction is done from the end to the beginning and all answer prefixes must be found in the answer_string.

    Example:
    This is a question. ① Yes ② No ③ Yes ④ No

    Expected output:
    [
        21,
        {"①": "Yes", "②": "No", "③": "Yes", "④": "No"},
    ]

    Args:
        answer_string (str): String possibly containing answers.
        answer_prefixes (list[str]): List of answer prefixes.
    Returns:
        Optional[tuple[int, dict[str, str]]]: A tuple containing the start index of the answer and dictionary mapping the prefix to the answer.
    """

    def extract_answer(acc: tuple[str, int, list[str]], symbol: str) -> tuple[str, int, list[str]]:
        """
        Extracts an answer from the text until the next symbol is found.
        Args:
            acc (tuple[str, int, list[str]]): Tuple containing the text, the right index (where to start searching) and the list of found answers.
            symbol (str): Symbol to extract the answer from.
        Returns:
            tuple[str, int, list[str]]: Tuple containing the text, the right index and the list of answers.
        """
        text, right_index, answers = acc
        if right_index == -1:
            return text, right_index, answers
        left_index = text.rfind(symbol, 0, right_index)
        if left_index == -1:
            return text, -1, answers
        return text, left_index, answers + [text[left_index:right_index]]

    # Try to extract answers from the possible_answers_part

    sorted_answer_prefixes = sorted(answer_prefixes, reverse=True)
    _, last_index, found_answers = reduce(
        extract_answer, sorted_answer_prefixes, (answer_string, len(answer_string), [])
    )
    if last_index == -1:
        return None

    # Ensure we have answers for all prefixes
    if len(found_answers) != len(answer_prefixes):
        return None

    found_answers = [answer.rstrip(PUNCT + ";").strip() for answer in found_answers]
    prefix_answer_dict = {
        answer[: len(prefix)]: answer[len(prefix) :].strip() for answer, prefix in zip(found_answers, answer_prefixes)
    }
    return last_index, prefix_answer_dict

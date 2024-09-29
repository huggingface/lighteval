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
    separator: str
    if variant == "COMMA":
        separator = f"{translation_literals.comma}{translation_literals.word_space}"
    elif variant == "NEW_LINE":
        separator = "\n"

    return separator.join(choices)


def multichoice_compose(
    choices: list[str],
    gold_idx: list[int],
    variant: MULTICHOICE_JOIN_VARIANT,
    translation_literals: TranslationLiterals,
):
    if len(gold_idx) == 1:
        return choices, gold_idx

    multichoice_joiner = partial(multichoice_join, variant=variant, translation_literals=translation_literals)

    new_choices = [multichoice_joiner([choices[i] for i in gold_idx])] + [
        choice for i, choice in enumerate(choices) if i not in gold_idx
    ]

    # All correct choices are now at 0
    return new_choices, [0]


def extract_answers_from_string(answer_string: str, answer_prefixes: list[str]) -> tuple[int, list[list[str]]] | None:
    """
    Attempts to extract answers from the answer_string. The answers are identified by being prefixed with answer prefixes.
    The extraction is done from the end to the beginning and all answer prefixes must be found in the answer_string.

    Example:
    This is a question. ① Yes ② No ③ Yes ④ No

    Expected output:
    [
        21,
        ["Yes", "No"],
        ["Yes", "No"],
    ]

    Args:
        answer_string (str): String possibly containing answers.
        answer_prefixes (list[str]): List of answer prefixes.
    Returns:
        Optional[tuple[int, list[list[str]]]]: A tuple containing the start index of the answers, list of list of answers (as answers can have multiple correct solutions).
    """

    def extract_answer(acc: tuple[str, int, list[str]], symbol: str) -> tuple[str, int, list[str]]:
        """
        Extracts an answer from the text until the next symbol is found.
        If the last index == -1 it means we failed to find the symbol.
        """
        text, last_index, answers = acc
        if last_index == -1:
            return text, last_index, answers
        start_index = last_index
        end_index = text.rfind(symbol[:last_index])
        if end_index == -1:
            return text, -1, answers
        return text, end_index, answers + [text[end_index:start_index]]

    # Try to extract answers from the possible_answers_part
    answer_string = answer_string.strip()

    # Split by ①②③④ to g
    _, last_index, found_answers = reduce(
        extract_answer, sorted(answer_prefixes, reverse=True), (answer_string, len(answer_string), [])
    )
    if last_index == -1:
        return None

    found_answers = [x for x in found_answers if x.strip() != ""]

    # Ensure we have answers for all prefixes
    if len(found_answers) != len(answer_prefixes):
        return None

    found_answers = [answer.rstrip(PUNCT + ";").strip() for answer in found_answers]
    letter_answer_dict = {answer[:1]: answer[1:].strip() for answer in found_answers}

    new_answer_list = [[letter_answer_dict.get(letter) for letter in answer] for answer in answer_prefixes]

    if any(any(a is None for a in l_ans) for l_ans in new_answer_list):
        return None

    return last_index, new_answer_list

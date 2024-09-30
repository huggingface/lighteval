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

from dataclasses import dataclass
from typing import Literal

from lighteval.tasks.default_prompts import INTEGER_INDICES, LETTER_INDICES
from lighteval.tasks.templates.utils.translation_literals import TranslationLiterals


ChoicePrefix = Literal["Letters", "NativeLetters", "Numbers"]


@dataclass
class MCFFormulation:
    """
    MCF Formulation
    Presenting the choices as A. B. C.
    The target is A, B, C

    Args:
        choice_prefix (ChoicePrefix, optional): The choice prefix to use for the task. Defaults to "Letters".
    """

    choice_prefix: ChoicePrefix = "Letters"
    name: str = "MCF"


@dataclass
class HybridFormulation:
    """
    Hybrid Formulation
    Presenting the choices as A. B. C.
    The target is then the answer itself not A, B, C

    Args:
        choice_prefix (ChoicePrefix, optional): The choice prefix to use for the task. Defaults to "Letters".
    """

    choice_prefix: ChoicePrefix = "Letters"
    name: str = "Hybrid"


@dataclass
class CFFormulation:
    """
    CF Formulation
    No choices are presented, the target is the answer itself
    """

    name: str = "CF"


Formulation = CFFormulation | HybridFormulation | MCFFormulation


def get_prefix(choice_prefix: ChoicePrefix, translation_literals: TranslationLiterals):
    if choice_prefix == "Letters":
        return LETTER_INDICES
    elif choice_prefix == "NativeLetters":
        return translation_literals.indices
    elif choice_prefix == "Numbers":
        return INTEGER_INDICES


def build_choices(
    choices: list[str],
    formulation: Formulation,
    translation_literals: TranslationLiterals,
    use_sentence_space: bool = True,
):
    """
    Builds a string version of the choices based on passed formulation for available options presentation.
    For Hybrid/MCF, the choices are presented as A. OptionA B. OptionB C. OptionC etc.
    For CF no choices are presented

    Args:
        choices (list[str]): List of choices to be presented.
        formulation (Formulation): The formulation to use for the task.
        translation_literals (TranslationLiterals): The translation literals scoped to required language.
        use_sentence_space (bool, optional): Whether to use sentence or word space in front of the choice.
            The same value should be passed to `build_answers` function to ensure consistent tokenization.

        Defaults to True.
    """
    if isinstance(formulation, CFFormulation):
        return None

    prefixes = get_prefix(formulation.choice_prefix, translation_literals)

    # Note the "answer_space" being variant before the actual answer key, this ensures consistent tokenization with answers
    answer_space = translation_literals.sentence_space if use_sentence_space else translation_literals.word_space
    if isinstance(formulation, MCFFormulation):
        options = "\n".join(
            [
                f"{answer_space}{prefixes[i]}{translation_literals.full_stop}{translation_literals.sentence_space}{c}"
                for i, c in enumerate(choices)
            ]
        )
    else:
        options = "\n".join(
            [
                f"{translation_literals.sentence_space}{prefixes[i]}{translation_literals.full_stop}{answer_space}{c}"
                for i, c in enumerate(choices)
            ]
        )
    return options


def build_answers(
    answers: list[str],
    formulation: Formulation,
    translation_literals: TranslationLiterals,
    use_sentence_space: bool = True,
) -> list[str]:
    """
    Builds a string version of the answers based on passed formulation.
    For MCF, the answers are presented as A, B, C etc.
    For Hybrid/CF, the answers are presented as the answer itself.

    Args:
        answers (list[str]): List of answers to be presented.
        formulation (Formulation): The formulation to use for the task.
        translation_literals (TranslationLiterals): The translation literals scoped to required language.
        use_sentence_space (bool, optional): Whether to use sentence or word space in front of the answer. Defaults to True.
            The same value should be passed to `build_choices` function to ensure consistent tokenization.
    """
    if isinstance(formulation, MCFFormulation):
        prefixes = get_prefix(formulation.choice_prefix, translation_literals)
        answers = [prefixes[i] for i in range(len(answers))]

    # Same tokenization as with answer key in options!
    answer_space = translation_literals.sentence_space if use_sentence_space else translation_literals.word_space
    return [f"{answer_space}{a}" for a in answers]

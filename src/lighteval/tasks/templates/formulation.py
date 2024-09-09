from dataclasses import dataclass
from typing import Literal

from lighteval.tasks.default_prompts import INTEGER_INDICES, LETTER_INDICES
from lighteval.tasks.templates.translation_literals import TranslationLiterals


ChoicePrefix = Literal["Letters", "NativeLetters", "Numbers"]


@dataclass
class MCFFormulation:
    choice_prefix: ChoicePrefix = "Letters"


@dataclass
class HybridFormulation:
    choice_prefix: ChoicePrefix = "Letters"


@dataclass
class CFFormulation:
    pass


Formulation = CFFormulation | HybridFormulation | MCFFormulation


def get_prefix(choice_prefix: ChoicePrefix, translation_literals: TranslationLiterals):
    if choice_prefix == "Letters":
        return LETTER_INDICES
    elif choice_prefix == "NativeLetters":
        return translation_literals.indices
    elif choice_prefix == "Numbers":
        return INTEGER_INDICES


def build_options(answers: list[str], formulation: Formulation, translation_literals: TranslationLiterals):
    if isinstance(formulation, CFFormulation):
        return None

    prefixes = get_prefix(formulation.choice_prefix, translation_literals)

    # Note the sentence space before each option, this ensures consistent tokenization with answers
    options = "\n".join([f"{translation_literals.sentence_space}{prefixes[i]}. {c}" for i, c in enumerate(answers)])
    return f"{options}"


def build_answers(
    answers: list[str], formulation: Formulation, translation_literals: TranslationLiterals
) -> list[str]:
    if isinstance(formulation, MCFFormulation):
        prefixes = get_prefix(formulation.choice_prefix, translation_literals)
        answers = [prefixes[i] for i in range(len(answers))]

    return [f"{translation_literals.sentence_space}{a}" for a in answers]

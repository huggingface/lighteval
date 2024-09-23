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

from typing import Callable, Literal

from typing_extensions import NotRequired, TypedDict

from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formatting_utils import PUNCT, capitalize, decapitalize, fix_ending_punct
from lighteval.tasks.templates.utils.formulation import CFFormulation, Formulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS, TranslationLiterals
from lighteval.tasks.templates.utils.utils import create_adapter_from_dict
from lighteval.utils.language import Language


NLI_TEMPLATE_QUERY_CF = "{instruction}{premise}{word_space}{confirmation_word}{question_mark}"
NLI_TEMPLATE_CONT_CF = "{sentence_space}{label}{comma}{word_space}{hypothesis}"


class NLIInput(TypedDict):
    premise: str
    hypothesis: str
    gold_idx: int
    instruction: NotRequired[str]


class NLIAdapter(TypedDict):
    premise: str
    hypothesis: str
    gold_idx: str
    instruction: NotRequired[str]


RelationType = Literal["entailment", "neutral", "contradiction"]


def get_nli_prompt_function_natural(
    language: Language,
    adapter: Callable[[dict], NLIInput] | NLIAdapter,
    relations: list[RelationType],
):
    """
    Create a templated prompt function for a Natural Language Inference (NLI) task.
    Example tasks:
    - ANLI
    - XNLI

    Format:
    Premies right? | (Yes,No,Also), hypothesis

    Args:
        language (Language): The language of the NLI task.
        adapter (Callable[[dict], NLIInput] | NLIAdapter): A function or dictionary to adapt the input data to the required NLIInput format.
            Must map data from the dataset row to the NLIInput format.
            Note: The label must be a index in the relations list, based on the correct relation between the premise and hypothesis.

        relations (list[Literal["entailment", "neutral", "contradiction"]]): The list of possible relations for the NLI task.
        formulation (MCFFormulation, optional): The formulation to use for multiple-choice questions. Defaults to MCFFormulation().

    Returns:
        Callable: A function that generates NLI prompts based on the given parameters.
    """

    def get_relation_label(label: RelationType, translation_literals: TranslationLiterals):
        if label == "entailment":
            return translation_literals.yes
        elif label == "contradiction":
            return translation_literals.no
        elif label == "neutral":
            return translation_literals.also

    translation_literals = TRANSLATION_LITERALS[language]
    labels = [capitalize(get_relation_label(label, translation_literals)) for label in relations]
    adapter_fn = create_adapter_from_dict(adapter) if isinstance(adapter, dict) else adapter  # type: ignore

    def nli_natural_prompt(line: dict, task_name: str):
        input_data = adapter_fn(line)
        premise, hypothesis, label = input_data["premise"], input_data["hypothesis"], input_data["gold_idx"]

        premise = capitalize(input_data["premise"].rstrip(PUNCT))
        hypothesis = decapitalize(input_data["hypothesis"].rstrip(PUNCT))

        query = NLI_TEMPLATE_QUERY_CF.format(
            instruction=input_data.get("instruction", ""),
            premise=premise,
            full_stop=translation_literals.full_stop,
            confirmation_word=translation_literals.confirmation_word,
            question_mark=translation_literals.question_mark,
            word_space=translation_literals.word_space,
        )

        choices = [
            NLI_TEMPLATE_CONT_CF.format(
                hypothesis=hypothesis,
                label=label,
                comma=translation_literals.comma,
                word_space=translation_literals.word_space,
                sentence_space=translation_literals.sentence_space,
            )
            for label in labels
        ]

        unconditioned_query = NLI_TEMPLATE_QUERY_CF.format(
            instruction="",
            premise="",
            confirmation_word=translation_literals.confirmation_word,
            question_mark=translation_literals.question_mark,
            word_space="",
            full_stop=translation_literals.full_stop,
        )

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=label,
            unconditioned_query=unconditioned_query,
        )

    return nli_natural_prompt


def get_nli_prompt_function(
    language: Language,
    adapter: Callable[[dict], NLIInput] | NLIAdapter,
    relations: list[RelationType],
    formulation: Formulation = MCFFormulation(),
):
    """
    Create a templated prompt function for a Natural Language Inference (NLI) task.
    Example tasks:
    - ANLI
    - XNLI

    Format:
    Premise
    Hypoothesis True,False,Neither? | (True,False,Neither)

    Args:
        language (Language): The language of the NLI task.
        adapter (Callable[[dict], NLIInput] | NLIAdapter): A function or dictionary to adapt the input data to the required NLIInput format.
            Must map data from the dataset row to the NLIInput format.
            Note: The gold_idx must be an index in the relations list, indicating the correct relation.
        relations (list[RelationType]): List of relation types to use for the task (e.g., ["entailment", "contradiction", "neutral"]).
        formulation (Formulation, optional): The formulation to use for the task. Defaults to MCFFormulation().

    Returns:
        Callable: A function that generates NLI prompts based on the given parameters.
    """
    translation_literals = TRANSLATION_LITERALS[language]

    def get_relation_label(label: RelationType, translation_literals: TranslationLiterals):
        if label == "entailment":
            return translation_literals.true
        elif label == "contradiction":
            return translation_literals.false
        elif label == "neutral":
            return translation_literals.neither

    labels = [capitalize(get_relation_label(label, translation_literals)) for label in relations]
    adapter_fn = create_adapter_from_dict(adapter) if isinstance(adapter, dict) else adapter  # type: ignore

    mcq_prompt_fn = get_mcq_prompt_function(
        language,
        {"context": "premise", "question": "hypothesis", "choices": "choices", "gold_idx": "gold_idx"},
        formulation,
    )

    def prompt_fn(line: dict, task_name: str):
        # Template based on dicussion here: https://github.com/EleutherAI/lm-evaluation-harness/issues/450
        input_data = adapter_fn(line)
        premise, hypothesis, gold_idx = input_data["premise"], input_data["hypothesis"], input_data["gold_idx"]
        premise = fix_ending_punct(capitalize(input_data["premise"]), translation_literals)
        hypothesis = input_data["hypothesis"]
        if isinstance(formulation, CFFormulation):
            # We inline the options into the hypothesis
            hypothesis = f"{hypothesis.rstrip(PUNCT)}{translation_literals.sentence_space}{capitalize(translation_literals.true)}{translation_literals.comma}{translation_literals.word_space}{capitalize(translation_literals.false)}{translation_literals.word_space}{translation_literals.or_word}{translation_literals.word_space}{capitalize(translation_literals.neither)}{translation_literals.question_mark}"

        row = {
            "instruction": input_data.get("instruction", ""),
            "premise": premise,
            "hypothesis": hypothesis,
            "gold_idx": gold_idx,
            "choices": labels,
        }

        return mcq_prompt_fn(row, task_name)

    return prompt_fn

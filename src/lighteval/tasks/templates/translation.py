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

from typing import Callable

from langcodes import standardize_tag
from typing_extensions import NotRequired, TypedDict

from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.multichoice import create_adapter_from_dict
from lighteval.tasks.templates.utils.formatting_utils import capitalize, fix_ending_punct
from lighteval.tasks.templates.utils.formulation import Formulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language
from lighteval.utils.utils import as_list


# Template chosen so that it's not very language-dependent, as it's not clear whether one should use the target or source language.
# It's also the best template based on https://arxiv.org/pdf/2301.07069.


TRANSLATION_CONTEXT = "{source_label}{colon}{sentence_space}{source_text}{sentence_space}{target_label}{colon}"


# Defined for type hinting only
class TranslationInput(TypedDict):
    """
    Input for the Translation task.
    Args:
        source_text: The source text to be translated
        target_text: The target text to be translated
        instruction (optional): The instruction of the Translation task (e.g. Translate the following text to Turkish)
    """

    source_text: str
    target_text: str | list[str]
    gold_idx: NotRequired[int | list[int]]
    instruction: NotRequired[str]


class TranslationAdapter(TypedDict):
    """
    Adapter for mapping from the dataset row into the TranslationInput format.
    Args:
        source_text: Column name in the row that contains the source text to be translated
        target_text: Column name in the row that contains the target text to be translated
        instruction (optional): Column name in the row that contains the instruction of the task (e.g. Translate the following text to Turkish)
    """

    source_text: str
    target_text: str
    gold_idx: NotRequired[int | list[int]]
    instruction: NotRequired[str]


def get_translation_prompt_function(
    source_language: Language,
    target_language: Language,
    adapter: Callable[[dict], TranslationInput | None] | TranslationAdapter,
    formulation: Formulation = MCFFormulation(),
):
    """
    Create a templated prompt function for a Translation task.
    Example tasks:
    - WMT2016
    - WMT2017

    Format:
    *CF*
    EN: How are you? TR: | Nasılsın?

    *Hybrid*
    EN: How are you? TR:
    A. Nasılsın?
    B. Jak se máš?
    Answer: | Nasılsın?/Jak se máš?

    *MCF*
    EN: How are you? TR:
    A. Nasılsın?
    B. Jak se máš?
    Answer: | A/B

    Args:
        adapter (Callable[[dict], TranslationInput] | TranslationAdapter): Either a function that takes a dataset row and returns a TranslationInput, or a dictionary with keys corresponding to the field names in the dataset row.
            Note: Both TranslationAdapter and TranslationInput are TypeDicts, this means that the caller provides dictionary and doesn't initialize any class!
        formulation (Formulation, optional): The formulation to use for the task. Defaults to MCFFormulation().
    Returns:
        Callable: A function that generates Translation prompts based on the given parameters.
    """
    adapter_fn = create_adapter_from_dict(adapter)
    continuation_prompt_fn = get_continuation_prompt_function(
        Language.ENGLISH,
        {"context": "context", "continuations": "continuations", "gold_idx": "gold_idx"},
        formulation,
        fix_formatting=False,
    )
    source_translation_literals = TRANSLATION_LITERALS[source_language]
    target_translation_literals = TRANSLATION_LITERALS[target_language]

    source_label_string = standardize_tag(source_language.value).upper()
    target_label_string = standardize_tag(target_language.value).upper()

    def translation_prompt(
        line: dict,
        task_name: str,
    ):
        input_data = adapter_fn(line)
        if input_data is None:
            return None

        source_text = capitalize(fix_ending_punct(input_data["source_text"], source_translation_literals))

        context = TRANSLATION_CONTEXT.format(
            source_label=source_label_string,
            source_text=source_text,
            target_label=target_label_string,
            colon=":",
            sentence_space=" ",
        )

        continuations = [
            capitalize(fix_ending_punct(text, target_translation_literals))
            for text in as_list(input_data["target_text"])
        ]

        return continuation_prompt_fn(
            {
                "instruction": input_data.get("instruction", ""),
                "context": context,
                "continuations": continuations,
                "gold_idx": input_data.get("gold_idx", list(range(len(continuations)))),
            },
            task_name,
        )

    return translation_prompt

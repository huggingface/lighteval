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

from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.multichoice import create_adapter_from_dict
from lighteval.tasks.templates.utils.formatting_utils import PUNCT, capitalize
from lighteval.tasks.templates.utils.formulation import Formulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


# NLI Cause/Effect (Copa)
COPA_QUERY = "{context}{word_space}{cause_or_effect}"


class COPAInput(TypedDict):
    cause_effect: Literal["cause", "effect"]
    context: str
    continuations: list[str]
    gold_idx: int | list[int]
    instruction: NotRequired[str]


class COPAAdapter(TypedDict):
    cause_effect: str
    context: str
    continuations: str
    gold_idx: str
    instruction: NotRequired[str]


def get_copa_prompt_function(
    language: Language, adapter: Callable[[dict], COPAInput] | COPAAdapter, formulation: Formulation = MCFFormulation()
):
    adapter_fn: Callable[[dict], COPAInput] = (
        create_adapter_from_dict(adapter) if isinstance(adapter, dict) else adapter
    )  # type: ignore
    continuation_prompt_fn = get_continuation_prompt_function(
        language, {"context": "context", "continuations": "continuations", "gold_idx": "gold_idx"}, formulation
    )
    translation_literals = TRANSLATION_LITERALS[language]

    def copa_prompt(
        line: dict,
        task_name: str,
    ):
        input_data = adapter_fn(line)
        context = capitalize(input_data["context"].rstrip(PUNCT))
        cause_or_effect_trans = (
            translation_literals.cause_word
            if input_data["cause_effect"] == "cause"
            else translation_literals.effect_word
        )

        context = COPA_QUERY.format(
            context=context,
            word_space=translation_literals.word_space,
            cause_or_effect=cause_or_effect_trans,
        )

        return continuation_prompt_fn(
            {
                "instruction": input_data.get("instruction", ""),
                "context": context,
                "continuations": input_data["continuations"],
                "gold_idx": input_data["gold_idx"],
            },
            task_name,
        )

    return copa_prompt

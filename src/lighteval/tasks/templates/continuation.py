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

from typing_extensions import NotRequired, TypedDict

from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.utils.formatting_utils import (
    capitalize,
    fix_capitalization,
    fix_ending_punct,
    should_follow_sentence_space,
)
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    Formulation,
    MCFFormulation,
    build_answers,
    build_options,
)
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.tasks.templates.utils.utils import create_adapter_from_dict
from lighteval.utils.language import Language
from lighteval.utils.utils import as_list


CONTINUATION_QUERY_CF = "{instruction}{context}"

CONTINUATION_QUERY_MCF = "{instruction}{context}\n{options}{answer_word}{colon}"


# NO idea how to ensure we have the same keys in these typedicts in python :(
class ContinuationInput(TypedDict):
    context: str
    continuations: list[str]
    gold_idx: list[int] | int
    instruction: NotRequired[str]


class ContinuationDictAdapter(TypedDict):
    context: str
    continuations: str
    gold_idx: str
    instruction: NotRequired[str]


# Python too dumb to do fancy inference :(


def get_continuation_prompt_function(
    language: Language,
    adapter: Callable[[dict], ContinuationInput] | ContinuationDictAdapter,
    formulation: Formulation = MCFFormulation(),
):
    adapter_fn: Callable[[dict], ContinuationInput] = (
        create_adapter_from_dict(adapter) if isinstance(adapter, dict) else adapter  # type: ignore
    )
    translation_literals = TRANSLATION_LITERALS[language]

    def prepare_prompt(line: dict):
        cont_input = adapter_fn(line)

        instruction_val = cont_input.get("instruction")
        instruction = f"{instruction_val}\n" if instruction_val else ""

        context = f"{capitalize(fix_ending_punct(cont_input['context'], translation_literals))}"

        continuations = [
            fix_capitalization(context, fix_ending_punct(continuation, translation_literals), translation_literals)
            for continuation in cont_input["continuations"]
        ]

        return cont_input, instruction, context, continuations

    def prompt_fn_cf(line, task_name: str):
        cont_input, instruction, context, continuations = prepare_prompt(line)

        context_follows_sentence_space = should_follow_sentence_space(context, translation_literals)
        answers = build_answers(continuations, formulation, translation_literals, context_follows_sentence_space)

        query = CONTINUATION_QUERY_CF.format(
            instruction=instruction,
            context=context,
        )

        return Doc(
            task_name=task_name,
            query=query,
            gold_index=as_list(cont_input["gold_idx"]),
            choices=answers,
            instruction=instruction,
            unconditioned_query="",
        )

    def prompt_fn_mcf(line, task_name: str):
        cont_input, instruction, context, continuations = prepare_prompt(line)

        options = build_options(continuations, formulation, translation_literals)
        options = f"{options}\n" if options else ""
        answers = build_answers(continuations, formulation, translation_literals)

        answer_word = capitalize(translation_literals.answer)

        query = CONTINUATION_QUERY_MCF.format(
            instruction=instruction,
            context=context,
            options=options,
            answer_word=answer_word,
            colon=translation_literals.colon,
        )

        return Doc(
            task_name=task_name,
            query=query,
            gold_index=as_list(cont_input["gold_idx"]),
            choices=answers,
            instruction=instruction,
            unconditioned_query=f"{answer_word}{translation_literals.colon}",
        )

    return prompt_fn_cf if isinstance(formulation, CFFormulation) else prompt_fn_mcf

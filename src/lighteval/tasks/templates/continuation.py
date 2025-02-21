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
from lighteval.tasks.templates.utils.adapter_utils import create_adapter_from_dict
from lighteval.tasks.templates.utils.formatting_utils import (
    capitalize,
    fix_capitalization,
    fix_ending_punct,
    punctuation_ends_sentence,
)
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    Formulation,
    MCFFormulation,
    build_answers,
    build_choices,
)
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language
from lighteval.utils.utils import as_list


CONTINUATION_QUERY_CF = "{instruction}{context}"

CONTINUATION_QUERY_MCF = "{instruction}{context}\n{options}{answer_word}{colon}"


# Defined for type hinting only
class ContinuationInput(TypedDict):
    """
    Input for the continuation task.
    Args:
        context: The contextualization of choices (e.g. If I ask you a question, you should answer it)
        continuations: Possible continuations of the context (e.g. [you should answer it, you should leave])
        gold_idx: The index of the correct continuation
        instruction (optional): The instruction of the task (e.g. Following is the snippet of a dialogue, choose the most appropriate continuation)
    """

    context: str
    continuations: list[str]
    gold_idx: list[int] | int
    instruction: NotRequired[str]


class ContinuationDictAdapter(TypedDict):
    """
    Adapter for mapping from the dataset row into the ContinuationInput format.
    Args:
        context: Column name in the row that contains the contextualization of choices (e.g. If I ask you a question, you should answer it)
        continuations: Column name in the row that contains the possible continuations of the context (e.g. [you should answer it, you should leave])
        gold_idx: Column name in the row that contains the index of the correct continuation
        instruction (optional): Column name in the row that contains the instruction of the task (e.g. Following is the snippet of a dialogue, choose the most appropriate continuation)
    """

    context: str
    continuations: str
    gold_idx: str
    instruction: NotRequired[str]


def get_continuation_prompt_function(
    language: Language,
    adapter: Callable[[dict], ContinuationInput | None] | ContinuationDictAdapter,
    formulation: Formulation = MCFFormulation(),
    fix_formatting: bool = True,
):
    """
    Create a templated prompt function for a Continuation task.
    Example tasks:
    - Hellaswag
    - XStoryCloze

    Format:
    *CF*
    Context | Continuation 1/Continuation 2/Continuation 3

    *Hybrid*
    Context
    A. Continuation 1
    B. Continuation 2
    C. Continuation 3
    Answer: Continuation 1/Continuation 2/Continuation 3

    *MCF*
    Context
    A. Continuation 1
    B. Continuation 2
    C. Continuation 3
    Answer: A/B/C

    This template is very similar to the `Multiple Choice` template, except that it only takes context/continuations as input and doesn't use the anchor labels (Question/Answer)

    Args:
        language (Language): The language of the Continuation task.
        adapter (Callable[[dict], ContinuationInput] | ContinuationDictAdapter): Either a function that takes a dataset row and returns a ContinuationInput, or a dictionary with keys corresponding to the field names in the dataset row.
            Note: Both ContinuationDictAdapter and ContinuationInput are TypeDicts, this means that the caller provides dictionary and doesn't initialize any class!
        formulation (Formulation, optional): The formulation (MCF/Hybrid/CF) to use for the task. Defaults to MCFFormulation().
        fix_formatting (bool, optional): Whether to fix the formatting of the text by capitalizing and fixing punctuation based on language. If False, the text will be used as-is. Defaults to True.
    Returns:
        Callable: A function that generates Continuation prompt based on the given parameters.
    """
    adapter_fn = create_adapter_from_dict(adapter)
    translation_literals = TRANSLATION_LITERALS[language]

    def prepare_prompt(line: dict):
        cont_input = adapter_fn(line)
        if cont_input is None:
            return None

        instruction_val = cont_input.get("instruction")
        instruction = f"{instruction_val}\n" if instruction_val else ""

        context = (
            f"{capitalize(fix_ending_punct(cont_input['context'], translation_literals))}"
            if fix_formatting
            else cont_input["context"]
        )

        continuations = [
            fix_capitalization(context, fix_ending_punct(continuation, translation_literals), translation_literals)
            if fix_formatting
            else continuation
            for continuation in cont_input["continuations"]
        ]

        return cont_input, instruction, context, continuations

    def prompt_fn_cf(line, task_name: str):
        prepared_prompt = prepare_prompt(line)
        if prepared_prompt is None:
            return None

        cont_input, instruction, context, continuations = prepared_prompt

        context_follows_sentence_space = punctuation_ends_sentence(context, translation_literals)
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
        prepared_prompt = prepare_prompt(line)
        if prepared_prompt is None:
            return None

        cont_input, instruction, context, continuations = prepared_prompt

        options = build_choices(continuations, formulation, translation_literals)
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

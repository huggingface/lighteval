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

from lighteval.tasks.default_prompts import hellaswag_preprocess
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.multichoice import create_adapter_from_dict
from lighteval.tasks.templates.utils.formatting_utils import (
    capitalize,
    fix_capitalization,
    fix_ending_punct,
    punctuation_ends_sentence,
)
from lighteval.tasks.templates.utils.formulation import Formulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


# NLI Cause/Effect (Copa)
HELLASWAG_QUERY = "{activity_label}{ctx}"


class HellaswagInput(TypedDict):
    ctx_a: str
    continuations: list[str]
    gold_idx: int | list[int]
    instruction: NotRequired[str]
    activity_label: NotRequired[str]
    ctx_b: NotRequired[str]


class HellaswagAdapter(TypedDict):
    ctx_a: str
    continuations: str
    gold_idx: str
    instruction: NotRequired[str]
    activity_label: NotRequired[str]
    ctx_b: NotRequired[str]


def get_hellaswag_prompt_function(
    language: Language,
    adapter: Callable[[dict], HellaswagInput | None] | HellaswagAdapter,
    formulation: Formulation = MCFFormulation(),
    wikihow_artifacts: list[str] = [" [title]"],
):
    """
    Create a templated prompt function for a Hellaswag task.

    Format:
    Context Premise therefore/cause | (Continuation 1, Continuation 2, Continuation 3)

    Args:
        language (Language): The language of the Hellaswag task.
        adapter (Callable[[dict], HellaswagInput] | HellaswagAdapter): A function or dictionary to adapt the input data to the required HellaswagInput format.
            Must map data from the dataset row to the HellaswagInput format.
            Note: The gold_idx must be an index or list of indices in the continuations list, indicating the correct continuation(s).
        formulation (Formulation, optional): The formulation to use for the task. Defaults to MCFFormulation().
        wikihow_artifacts (list[str], optional): A list of strings to replace with dot. We have to replace the the texts with dots because
            of wikihow source.

    Returns:
        Callable: A function that generates COPA prompts based on the given parameters.
    """

    translation_literals = TRANSLATION_LITERALS[language]

    def process_context(ctx):
        if ctx == "":
            return ""
        return capitalize(
            fix_ending_punct(
                hellaswag_preprocess(ctx, truncate_dots=True, wikihow_artifacts=wikihow_artifacts, strip_text=True),
                translation_literals,
            )
        )

    def join_ctxs(ctx_a, ctx_b):
        space = (
            translation_literals.sentence_space
            if punctuation_ends_sentence(ctx_a, translation_literals)
            else translation_literals.word_space
        )
        return f"{ctx_a.rstrip()}{space}{fix_capitalization(ctx_a, ctx_b, translation_literals)}"

    adapter_fn = create_adapter_from_dict(adapter)
    continuation_prompt_fn = get_continuation_prompt_function(
        language, {"context": "context", "continuations": "continuations", "gold_idx": "gold_idx"}, formulation
    )

    def hellaswag_prompt(
        line: dict,
        task_name: str,
    ):
        input_data = adapter_fn(line)
        if input_data is None:
            return None

        activity_label = input_data.get("activity_label", "")
        activity_label = f"{capitalize(activity_label)}:\n" if activity_label else ""

        # Last one should be left as is
        ctx_a, ctx_b = process_context(input_data["ctx_a"]), process_context(input_data.get("ctx_b", ""))
        if ctx_b:
            ctx_a = join_ctxs(ctx_a, ctx_b)

        # Removal of the [header] can happen and we need the first letter to be capital afterwards
        full_context = HELLASWAG_QUERY.format(activity_label=activity_label, ctx=ctx_a)
        choices = [
            hellaswag_preprocess(
                continuation,
                wikihow_artifacts=wikihow_artifacts,
                truncate_dots=True,
                strip_text=True,
                dot_replacement=f"{translation_literals.full_stop}{translation_literals.sentence_space}",
            )
            for continuation in input_data["continuations"]
        ]

        # It can happen that the continuations are empty we thus skip the task
        if any(len(c.strip()) == 0 for c in choices):
            return None

        return continuation_prompt_fn(
            {
                "instruction": input_data.get("instruction", ""),
                "context": full_context,
                "continuations": choices,
                "gold_idx": input_data["gold_idx"],
            },
            task_name,
        )

    return hellaswag_prompt

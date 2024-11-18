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

from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.adapter_utils import create_adapter_from_dict
from lighteval.tasks.templates.utils.formulation import Formulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


# Defined for type hinting only
class BoolQInput(TypedDict):
    """
    Input for the BoolQ task.
    Args:
        question: The question of the BoolQ task (e.g. What is the capital of Germany?)
        answer: The answer of the BoolQ task (True = yes, False = no)
        instruction (optional): The instruction of the BoolQ task (e.g. Choose the most appropriate continuation)
        context: The context of the BoolQ task (e.g. Munich is not capital of Germany)
    """

    question: str
    answer: bool
    instruction: NotRequired[str]
    context: NotRequired[str]


class BoolQAdapter(TypedDict):
    """
    Adapter for mapping from the dataset row into the BoolQInput format.
    Args:
        question: Column name in the row that contains the question of the BoolQ task (e.g. What is the capital of Germany?)
        answer: Column name in the row that contains the answer of the BoolQ task (True = yes, False = no)
        instruction (optional): Column name in the row that contains the instruction of the task (e.g. Choose the most appropriate continuation)
        context: Column name in the row that contains the context of the BoolQ task (e.g. Munich is not capital of Germany)
    """

    question: str
    answer: str
    instruction: NotRequired[str]
    context: NotRequired[str]


def get_boolq_prompt_function(
    language: Language,
    adapter: Callable[[dict], BoolQInput | None] | BoolQAdapter,
    formulation: Formulation = MCFFormulation(),
):
    """
    Create a templated prompt function for a BoolQ task.
    It leverages the translation literals (yes/no) for the choices. All other logic is the same as the mcq prompt function.
    Example tasks:
    - boolQ
    - acva

    Format:
    See mcq prompt function for the format.
    """
    translation_literals = TRANSLATION_LITERALS[language]

    adapter_fn = create_adapter_from_dict(adapter)
    mcq_prompt_fn = get_mcq_prompt_function(
        language,
        {
            "question": "question",
            "choices": "choices",
            "context": "context",
            "instruction": "instruction",
            "gold_idx": "gold_idx",
        },
        formulation,
    )

    def boolq_prompt(
        line: dict,
        task_name: str,
    ):
        input_data = adapter_fn(line)
        if input_data is None:
            return None

        choices = [translation_literals.yes, translation_literals.no]
        return mcq_prompt_fn(
            {
                "instruction": input_data.get("instruction", ""),
                "question": input_data["question"],
                "context": input_data.get("context", ""),
                "choices": choices,
                "gold_idx": 0 if input_data["answer"] else 1,
            },
            task_name,
        )

    return boolq_prompt

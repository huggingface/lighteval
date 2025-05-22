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

import logging
from typing import Callable

from lighteval.tasks.templates.multichoice import MCQInput, create_adapter_from_dict, get_mcq_prompt_function
from lighteval.tasks.templates.qa import QAAdapter, QAInput
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


logger = logging.get_logger(__name__)

WARNED_ABOUT_COT_INSTRUCTION = False


def get_math_qa_prompt_function(
    language: Language, adapter: Callable[[dict], QAInput | None] | QAAdapter, cot: bool = False
):
    """
    Create a templated prompt function for a MathQA task.
    Example tasks:
    - MGSM
    - GSM8K

    Format:
    Question: xxx
    Answer: | Answer

    Args:
        language (Language): The language of the MathQA task.
        adapter (Callable[[dict], QAInput] | QAAdapter): A function or dictionary to adapt the input data to the required QAInput format.
            Must map data from the dataset row to the QAInput format.

    Returns:
        Callable: A function that generates MathQA prompts based on the given parameters.
    """

    adapter_fn = create_adapter_from_dict(adapter)

    def adapter_for_mcq(line: dict) -> MCQInput | None:
        input_data = adapter_fn(line)
        if input_data is None:
            return None

        choices = input_data["choices"]
        translation_literals = TRANSLATION_LITERALS[language]

        instruction_val = input_data.get("instruction")
        if cot and not instruction_val:
            instruction_val = (
                f"{translation_literals.qa_instruction}\n{translation_literals.math_formatting_instruction}"
            )
            if not WARNED_ABOUT_COT_INSTRUCTION:
                logger.warning(
                    f" You are using a COT with MCF formulation but did not provide an instruction. Defaulting to {instruction_val}"
                )
                WARNED_ABOUT_COT_INSTRUCTION = True

        instruction = f"{instruction_val}\n\n" if instruction_val else ""

        return {
            **input_data,
            "gold_idx": list(range(len(choices))),
            "instruction": instruction,
        }

    multichoice_prompt_fn = get_mcq_prompt_function(
        language, adapter=adapter_for_mcq, formulation=CFFormulation(cot=cot)
    )
    return multichoice_prompt_fn

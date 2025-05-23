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


from lighteval.tasks.templates.qa import (
    QAInput,  # Reusing QAInput for simplicity
    get_qa_prompt_function,
)
from lighteval.utils.language import Language


def test_qa_prompt_cf():
    """
    Tests that QA prompt function works correctly for CF formulation.
    """
    test_input = {
        "question": "What is 2 + 2?",
        "choices": ["4", "5", "6"],
    }

    prompt_fn = get_qa_prompt_function(
        language=Language.ENGLISH,
        adapter=lambda x: QAInput(
            question=x["question"],
            choices=x["choices"],
        ),
        cot=False,
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc is not None

    assert (
        doc.query
        == """\
Question: What is 2 + 2?
Answer:\
"""
    )
    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" 4", " 5", " 6"]
    assert doc.gold_index == [0, 1, 2]


def test_qa_prompt_cf_cot_default_instruction():
    """
    Tests QA with CoT and default instruction, checking for the warning.
    """
    test_input = {
        "question": "Solve for x: x + 5 = 10",
        "choices": ["5", "10", "0"],
    }

    prompt_fn = get_qa_prompt_function(
        language=Language.ENGLISH,
        adapter=lambda x: QAInput(
            question=x["question"],
            choices=x["choices"],
        ),
        cot=True,
    )

    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == """\
Answer the following question.
Output the final answer in format: <b></b>.

Question: Solve for x: x + 5 = 10
Step-by-Step Answer:\
"""
    )
    assert doc.unconditioned_query == "Step-by-Step Answer:"
    assert doc.choices == [" 5", " 10", " 0"]
    assert doc.gold_index == [0, 1, 2]


def test_qa_prompt_cf_cot_custom_instruction():
    """
    Tests QA with CoT and custom instruction.
    """
    test_input = {
        "question": "Solve for x: x + 5 = 10",
        "choices": ["5", "10", "0"],
        "instruction": "Answer the following question. Output the final answer in format: <b></b>.",
        "few_shot_cot": "1+2+3+4+5=15",
        "__few_shots": True,
    }

    prompt_fn = get_qa_prompt_function(
        language=Language.ENGLISH,
        adapter=lambda x: QAInput(
            question=x["question"],
            choices=x["choices"],
            instruction=x["instruction"],
            few_shot_cot=x["few_shot_cot"],
        ),
        cot=True,
    )

    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == """\
Answer the following question. Output the final answer in format: <b></b>.

Question: Solve for x: x + 5 = 10
Step-by-Step Answer:\
"""
    )
    assert doc.unconditioned_query == "Step-by-Step Answer:"
    assert doc.choices == [" 1+2+3+4+5=15"]
    assert doc.gold_index == [0]

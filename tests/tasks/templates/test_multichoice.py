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

from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.utils.language import Language


def test_multichoice_prompt_mcf():
    # Define test input
    test_input = {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "gold_idxs": 1,
        "context": "France is a country in Western Europe.",
        "instruction": "Please answer the following question about geography.",
    }

    # Generate prompt using mcq_prompt_functions
    prompt_fn = get_mcq_prompt_function(
        Language.english,
        {
            "question": "question",
            "choices": "choices",
            "gold_idxs": "gold_idxs",
            "context": "context",
            "instruction": "instruction",
        },
        MCFFormulation(),
    )

    # Test mcq_prompt_functions directly
    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == """\
Please answer the following question about geography.
France is a country in Western Europe.
Question: What is the capital of France?
 A. London
 B. Paris
 C. Berlin
 D. Madrid
Answer:\
"""
    )

    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B", " C", " D"]


def test_multichoice_prompt_nli_cf():
    """
    Test that the NLI prompt with CF formulation works.
    """

    # Define test input
    test_input = {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "gold_idxs": 1,
        "context": "France is a country in Western Europe.",
        "instruction": "Please answer the following question about geography.",
    }

    # Test mcq_prompt_functions directly

    # Generate prompt using mcq_prompt_functions
    prompt_fn = get_mcq_prompt_function(
        Language.english,
        {
            "question": "question",
            "choices": "choices",
            "gold_idxs": "gold_idxs",
            "context": "context",
            "instruction": "instruction",
        },
        CFFormulation(),
    )
    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == """\
Please answer the following question about geography.
France is a country in Western Europe.
Question: What is the capital of France?
Answer:\
"""
    )

    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" London", " Paris", " Berlin", " Madrid"]


def test_chinese_multichoice_prompt():
    """
    Test that the multichoice prompt works for Chinese.
    This means that:
    - No spaces are used for separating words/sequneces
    - Correct punctuation characters are used
    """

    # Define test input
    test_input = {
        "question": "什么是中国的首都?",
        "choices": ["北京", "上海", "广州", "深圳"],
        "gold_idxs": 0,
    }

    prompt_fn = get_mcq_prompt_function(
        Language.chinese, {"question": "question", "choices": "choices", "gold_idxs": "gold_idxs"}, MCFFormulation()
    )

    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == """\
问题：什么是中国的首都？
A。北京
B。上海
C。广州
D。深圳
答案：\
"""
    )


def test_thai_multichoice_prompt():
    """
    Test that the multichoice prompt works for Thai.
    This means that:
    - No spaces are used for separating words
    """

    # Define test input
    test_input = {
        "question": "สิ่งใดต่อไปนี้เป็นสิ่งที่คุณชอบมากที่สุด?",
        "choices": ["รถยนต์", "รถจักรยานยนต์", "รถจักรยานยนต์", "รถยนต์"],
        "gold_idxs": 0,
    }

    prompt_fn = get_mcq_prompt_function(
        Language.thai, {"question": "question", "choices": "choices", "gold_idxs": "gold_idxs"}, MCFFormulation()
    )

    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == """\
คำถาม: สิ่งใดต่อไปนี้เป็นสิ่งที่คุณชอบมากที่สุด?
 A. รถยนต์
 B. รถจักรยานยนต์
 C. รถจักรยานยนต์
 D. รถยนต์
คำตอบ:\
"""
    )

    assert doc.unconditioned_query == "คำตอบ:"
    assert doc.choices == [" A", " B", " C", " D"]


def test_multichoice_optional_keys():
    """
    Test that the multichoice prompt works when the input dict has optional keys.
    - context
    - instruction
    """

    # Define test input with all keys
    test_input = {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "context": "France is big.",
        "instruction": "Please answer the following question about geography.",
        "gold_idxs": 1,
    }

    prompt_fn = get_mcq_prompt_function(
        Language.english,
        {
            "question": "question",
            "choices": "choices",
            "gold_idxs": "gold_idxs",
            "context": "context",
            "instruction": "instruction",
        },
        MCFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == """\
Please answer the following question about geography.
France is big.
Question: What is the capital of France?
 A. London
 B. Paris
 C. Berlin
 D. Madrid
Answer:\
"""
    )

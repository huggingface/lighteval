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


import pytest

from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.utils.language import Language


def test_translation_prompt_cf():
    """
    Tests that translation prompt function works correctly for CF formulation.
    """
    test_input = {
        "source_text": "Ahoj, jak se máš?",
        "target_text": "Bonjour, comment allez-vous?",
    }

    prompt_fn = get_translation_prompt_function(
        source_language=Language.CZECH,
        target_language=Language.FRENCH,
        adapter=lambda x: {
            "source_text": x["source_text"],
            "target_text": x["target_text"],
        },
        formulation=CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc is not None

    assert doc.query == "CS: Ahoj, jak se máš? FR:"
    assert doc.unconditioned_query == ""
    assert doc.choices == [" Bonjour, comment allez-vous?"]
    assert doc.gold_index == [0]


def test_translation_prompt_mcf():
    """
    Tests that translation prompt function works correctly for MCF formulation.
    """
    test_input = {
        "source_text": "Ahoj, jak se máš?",
        "target_text": ["Bonjour, comment allez-vous?", "Ciao, come stai?"],
    }

    prompt_fn = get_translation_prompt_function(
        source_language=Language.CZECH,
        target_language=Language.FRENCH,
        adapter=lambda x: {
            "source_text": x["source_text"],
            "target_text": x["target_text"],
            "gold_idx": 0,
        },
        formulation=MCFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc is not None

    assert (
        doc.query
        == """\
CS: Ahoj, jak se máš? FR:

Options:
 A. Bonjour, comment allez-vous?
 B. Ciao, come stai?
Answer:\
"""
    )
    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B"]
    assert doc.gold_index == [0]


def test_translation_prompt_cf_formatting():
    """
    Tests that translation prompt function works correctly for CF formulation with formatting.
    """
    test_input = {
        "source_text": "How are you?",
        "target_text": ["你好吗?"],
    }

    prompt_fn = get_translation_prompt_function(
        source_language=Language.ENGLISH,
        target_language=Language.CHINESE,
        adapter=lambda x: {
            "source_text": x["source_text"],
            "target_text": x["target_text"],
            "gold_idx": 0,
        },
        formulation=CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc is not None

    assert doc.query == "EN: How are you? ZH:"
    assert doc.unconditioned_query == ""
    assert doc.choices == [" 你好吗？"]
    assert doc.gold_index == [0]


def test_translation_cot_default_instruction():
    """
    Tests that translation prompt function uses default instruction when CoT is set to true.
    """
    test_input = {
        "source_text": "How are you?",
        "target_text": "你好吗?",
    }

    prompt_fn = get_translation_prompt_function(
        source_language=Language.ENGLISH,
        target_language=Language.CHINESE,
        adapter=lambda x: {
            "source_text": x["source_text"],
            "target_text": x["target_text"],
        },
        formulation=CFFormulation(cot=True),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc is not None

    # Check that the default instruction is included
    expected_instruction = "Translate the following text from English to Chinese.\n"
    assert doc.query.startswith(expected_instruction)
    assert "EN: How are you? ZH:" in doc.query
    assert doc.choices == [" 你好吗？"]
    assert doc.gold_index == [0]


def test_translation_cot_default_instruction_mcf():
    """
    Tests that translation prompt function uses default instruction when CoT is set to true for MCF formulation.
    """
    test_input = {
        "source_text": "Ahoj, jak se máš?",
        "target_text": ["Bonjour, comment allez-vous?", "Ciao, come stai?"],
    }

    prompt_fn = get_translation_prompt_function(
        source_language=Language.CZECH,
        target_language=Language.FRENCH,
        adapter=lambda x: {
            "source_text": x["source_text"],
            "target_text": x["target_text"],
            "gold_idx": 0,
        },
        formulation=MCFFormulation(cot=True),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc is not None

    # Check that both default instructions are included
    expected_instructions = (
        "Choose the letter of the correct answer.\nOutput the final answer in format: <b></b>.\n\n"
    )
    assert doc.query.startswith(expected_instructions)
    assert "CS: Ahoj, jak se máš? FR:" in doc.query
    assert "A. Bonjour, comment allez-vous?" in doc.query
    assert "B. Ciao, come stai?" in doc.query
    assert doc.choices == [" A", " B"]
    assert doc.gold_index == [0]


def test_translation_cot_user_instruction():
    """
    Tests that translation prompt function uses user provided instruction when available.
    """
    test_input = {
        "source_text": "How are you?",
        "target_text": "你好吗?",
        "instruction": "Please translate this English text to Chinese:",
    }

    prompt_fn = get_translation_prompt_function(
        source_language=Language.ENGLISH,
        target_language=Language.CHINESE,
        adapter=lambda x: {
            "source_text": x["source_text"],
            "target_text": x["target_text"],
            "instruction": x["instruction"],
        },
        formulation=CFFormulation(cot=True),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc is not None

    # Check that the user instruction is included with formatting instruction
    expected_instructions = (
        "Please translate this English text to Chinese:\n\n"
    )
    assert doc.query.startswith(expected_instructions)
    assert "EN: How are you? ZH:" in doc.query
    assert doc.choices == [" 你好吗？"]
    assert doc.gold_index == [0]


def test_translation_cot_mcf_number_prefix_error():
    """
    Tests that translation prompt function raises an error when using CoT with MCF and Number choice prefix.
    """
    test_input = {
        "source_text": "How are you?",
        "target_text": "你好吗?",
    }

    with pytest.raises(ValueError, match="You are using a COT with a unsupported formulation"):
        prompt_fn = get_translation_prompt_function(
            source_language=Language.ENGLISH,
            target_language=Language.CHINESE,
            adapter=lambda x: {
                "source_text": x["source_text"],
                "target_text": x["target_text"],
                "gold_idx": 0,
            },
            formulation=MCFFormulation(cot=True, choice_prefix="Numbers"),
        )

        prompt_fn(test_input, "test_task")


def test_translation_prompt_mcf_cot():
    """
    Tests that translation prompt function works correctly for both cause/effect.
    Since it's pretty much a wrapper around continuation template we just test single formulation.

    """
    test_input = {
        "source_text": "How are you?",
        "target_text": ["你好吗?", "你怎么样?"],
        "__few_shots": True,
        "few_shot_cot": "i think it's A.",
        "gold_idx": 0,
        "instruction": "Choose the letter of the most likely continuation.",
    }

    prompt_fn = get_translation_prompt_function(
        Language.ENGLISH,
        Language.CHINESE,
        {
            "source_text": "source_text",
            "target_text": "target_text",
            "gold_idx": "gold_idx",
            "few_shot_cot": "few_shot_cot",
            "instruction": "instruction",
        },
        MCFFormulation(cot=True),
    )

    doc = prompt_fn(test_input, "test_task")

    assert (
        doc.query
        == f"""\
Choose the letter of the most likely continuation.

EN: How are you? ZH:

Options:
 A. 你好吗？
 B. 你怎么样？
Step-by-Step Answer:\
"""
    )

    assert doc.unconditioned_query == "Step-by-Step Answer:"
    assert doc.choices == [" I think it's A."]
    assert doc.gold_index == [0]

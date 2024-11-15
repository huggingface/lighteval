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

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


from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.utils.language import Language


def test_hellaswag_prompt_cf():
    """
    Tests that hellaswag prompt function works correctly.
    Since it's pretty much a wrapper around continuation template we just test single formulation.

    """
    test_input = {
        "activity_label": "fitness",
        "ctx_a": "He is strong",
        "ctx_b": "He is fast",
        "continuations": ["he has big muscles", "he is weak"],
        "gold_idx": 0,
    }

    prompt_fn = get_hellaswag_prompt_function(
        Language.ENGLISH,
        {
            "activity_label": "activity_label",
            "continuations": "continuations",
            "gold_idx": "gold_idx",
            "ctx_a": "ctx_a",
            "ctx_b": "ctx_b",
        },
        CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc.query == "Fitness:\nHe is strong he is fast"

    assert doc.unconditioned_query == ""
    assert doc.choices == [" he has big muscles", " he is weak"]
    assert doc.gold_index == [0]


def test_hellaswag_prompt_mcf():
    """
    Tests that hellaswag prompt function works correctly.
    Since it's pretty much a wrapper around continuation template we just test single formulation.

    """
    test_input = {
        "activity_label": "fitness",
        "ctx_a": "He is strong",
        "ctx_b": "He is fast",
        "continuations": ["he has big muscles", "he is weak"],
        "gold_idx": 0,
    }

    prompt_fn = get_hellaswag_prompt_function(
        Language.ENGLISH,
        {
            "activity_label": "activity_label",
            "continuations": "continuations",
            "gold_idx": "gold_idx",
            "ctx_a": "ctx_a",
            "ctx_b": "ctx_b",
        },
        MCFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")
    assert (
        doc.query
        == """\
Fitness:\nHe is strong he is fast
 A. he has big muscles
 B. he is weak
Answer:\
"""
    )

    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B"]
    assert doc.gold_index == [0]


def test_hellaswag_ctx_joining():
    """
    Tests that hellaswag prompt function works correctly.
    Since it's pretty much a wrapper around continuation template we just test single formulation.

    """
    test_input = {
        "activity_label": "fitness",
        "ctx_a": "He is strong.",
        "ctx_b": "he is fast.",
        "continuations": ["he has big muscles", "he is weak"],
        "gold_idx": 0,
    }

    prompt_fn = get_hellaswag_prompt_function(
        Language.ENGLISH,
        {
            "activity_label": "activity_label",
            "continuations": "continuations",
            "gold_idx": "gold_idx",
            "ctx_a": "ctx_a",
            "ctx_b": "ctx_b",
        },
        CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc.query == "Fitness:\nHe is strong. He is fast."


def test_hellaswag_single_ctx():
    """
    Tests that hellaswag prompt function works correctly.
    Since it's pretty much a wrapper around continuation template we just test single formulation.

    """
    test_input = {
        "activity_label": "fitness",
        "ctx_a": "He is strong.",
        "continuations": ["he has big muscles", "he is weak"],
        "gold_idx": 0,
    }

    prompt_fn = get_hellaswag_prompt_function(
        Language.ENGLISH,
        {
            "activity_label": "activity_label",
            "continuations": "continuations",
            "gold_idx": "gold_idx",
            "ctx_a": "ctx_a",
        },
        CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")
    assert doc.query == "Fitness:\nHe is strong."

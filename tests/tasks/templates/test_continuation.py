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

from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.utils.language import Language


def test_continuation_prompt_mcf():
    # Define test input
    test_input = {
        "context": "The quick brown fox",
        "continuations": ["jumps over the lazy dog", "Runs through the forest", "Chases a rabbit"],
        "gold_idxs": 0,
    }

    # Generate prompt using continuation_prompt_function
    prompt_fn = get_continuation_prompt_function(
        Language.english,
        {"context": "context", "continuations": "continuations", "gold_idx": "gold_idx"},
        MCFFormulation(),
    )

    # Test continuation_prompt_function
    doc = prompt_fn(test_input, "test_continuation_task")

    assert (
        doc.query
        == """\
The quick brown fox
 A. jumps over the lazy dog
 B. runs through the forest
 C. chases a rabbit
Answer:\
"""
    )

    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B", " C"]
    assert doc.gold_index == [0]


def test_continuation_prompt_cf():
    # Define test input
    test_input = {
        "context": "The sun is",
        "continuations": ["shining brightly", "setting in the west", "hidden behind clouds"],
        "gold_idxs": 1,
    }

    # Generate prompt using continuation_prompt_function with CF formulation
    prompt_fn = get_continuation_prompt_function(
        Language.english,
        {"context": "context", "continuations": "continuations", "gold_idx": "gold_idx"},
        CFFormulation(),
    )

    # Test continuation_prompt_function
    doc = prompt_fn(test_input, "test_continuation_task")

    assert doc.query == "The sun is"

    assert doc.unconditioned_query == ""
    assert doc.choices == [" shining brightly", " setting in the west", " hidden behind clouds"]
    assert doc.gold_index == [1]


def test_continuation_prompt_sequence_end():
    """
    Test that the continuations are properly adjusted when the context is a finite sequence.
    """
    test_input = {
        "context": "the sun is.",
        "continuations": ["shining brightly", "setting in the west", "hidden behind clouds"],
        "gold_idxs": 1,
    }

    prompt_fn = get_continuation_prompt_function(
        Language.english,
        {"context": "context", "continuations": "continuations", "gold_idx": "gold_idx"},
        CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_continuation_task")

    assert doc.query == "The sun is."

    assert doc.unconditioned_query == ""
    assert doc.choices == [" Shining brightly", " Setting in the west", " Hidden behind clouds"]
    assert doc.gold_index == [1]


def test_continuation_optional_keys():
    """
    Test that the continuation_prompt_function can handle optional keys:
    - instruction
    """
    # Define test input with optional keys
    test_input = {
        "context": "In the morning, I like to",
        "continuations": ["drink coffee", "go for a run", "read the news"],
        "gold_idxs": 0,
        "instruction": "Choose the most likely continuation:",
    }

    # Generate prompt using continuation_prompt_function with optional keys
    prompt_fn = get_continuation_prompt_function(
        Language.english,
        {
            "context": "context",
            "continuations": "continuations",
            "gold_idx": "gold_idx",
            "instruction": "instruction",
        },
        MCFFormulation(),
    )

    # Test continuation_prompt_function
    doc = prompt_fn(test_input, "test_continuation_task")

    assert (
        doc.query
        == """\
Choose the most likely continuation:
In the morning, I like to
 A. drink coffee
 B. go for a run
 C. read the news
Answer:\
"""
    )

    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B", " C"]
    assert doc.gold_index == [0]

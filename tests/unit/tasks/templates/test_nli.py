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

from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation
from lighteval.utils.language import Language


def test_nli_prompt_mcf():
    """Test multiple-choice format NLI prompt generation."""
    test_input = {
        "premise": "The cat is sleeping on the couch.",
        "hypothesis": "The cat is awake.",
        "gold_idx": 2,
    }

    prompt_fn = get_nli_prompt_function(
        Language.ENGLISH,
        {"hypothesis": "hypothesis", "premise": "premise", "gold_idx": "gold_idx"},
        ["entailment", "neutral", "contradiction"],
    )

    doc = prompt_fn(test_input, "test_nli_task")

    assert (
        doc.query
        == """\
The cat is sleeping on the couch.
Question: The cat is awake.
 A. True
 B. Neither
 C. False
Answer:\
"""
    )
    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B", " C"]
    assert doc.gold_index == [2]


def test_nli_prompt_cf():
    """Test cloze format NLI prompt generation."""
    test_input = {
        "premise": "The cat is sleeping on the couch.",
        "hypothesis": "The cat is awake.",
        "gold_idx": 2,
    }

    prompt_fn = get_nli_prompt_function(
        Language.ENGLISH,
        {"hypothesis": "hypothesis", "premise": "premise", "gold_idx": "gold_idx"},
        ["entailment", "neutral", "contradiction"],
        formulation=CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_nli_task")

    assert doc.query == "The cat is sleeping on the couch right?"
    assert doc.unconditioned_query == "right?"
    assert doc.choices == [" Yes, the cat is awake", " Also, the cat is awake", " No, the cat is awake"]
    assert doc.gold_index == 2

    test_input = {
        "premise": "The cat is sleeping on the couch.",
        "hypothesis": "The cat is awake.",
        "gold_idx": 2,
    }


def test_nli_prompt_hybrid():
    """Test hybrid format NLI prompt generation."""

    test_input = {
        "premise": "The cat is sleeping on the couch.",
        "hypothesis": "The cat is awake.",
        "gold_idx": 2,
    }
    prompt_fn = get_nli_prompt_function(
        Language.ENGLISH,
        {"hypothesis": "hypothesis", "premise": "premise", "gold_idx": "gold_idx"},
        ["entailment", "neutral", "contradiction"],
        formulation=HybridFormulation(),
    )

    doc = prompt_fn(test_input, "test_nli_task")

    assert (
        doc.query
        == """\
The cat is sleeping on the couch.
Question: The cat is awake True, False or Neither?
Answer:\
"""
    )
    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" True", " Neither", " False"]
    assert doc.gold_index == [2]

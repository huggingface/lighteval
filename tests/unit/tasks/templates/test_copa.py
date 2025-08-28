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

from lighteval.tasks.templates.copa import get_copa_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language


@pytest.mark.parametrize("cause_effect", ["cause", "effect"])
def test_copa_prompt_cf(cause_effect):
    """
    Tests that copa prompt function works correctly for both cause/effect.
    Since it's pretty much a wrapper around continuation template we just test single formulation.

    """
    test_input = {
        "cause_effect": cause_effect,
        "context": "He is strong",
        "continuations": ["he has big muscles", "he is weak"],
        "gold_idx": 0,
    }

    prompt_fn = get_copa_prompt_function(
        Language.ENGLISH,
        {
            "cause_effect": "cause_effect",
            "context": "context",
            "continuations": "continuations",
            "gold_idx": "gold_idx",
        },
        CFFormulation(),
    )

    doc = prompt_fn(test_input, "test_task")

    cause_effect_word = "because" if cause_effect == "cause" else "therefore"
    assert doc.query == f"He is strong {cause_effect_word}"

    assert doc.unconditioned_query == ""
    assert doc.choices == [" he has big muscles", " he is weak"]
    assert doc.gold_index == [0]

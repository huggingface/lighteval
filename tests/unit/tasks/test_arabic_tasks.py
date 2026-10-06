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

import sys

import pytest

from lighteval.tasks.multilingual.tasks.arabic import hellaswag_arabic_pfn


def test_hellaswag_arabic_pfn_parses_endings():
    line = {
        "ctx": "sentence one [latin] and more",
        "endings": "['ending one [x]', 'ending two']",
        "label": 1,
    }

    doc = hellaswag_arabic_pfn(line, "test_task")

    assert doc.choices == ["ending one ", "ending two"]
    assert doc.gold_index == 1


def test_hellaswag_arabic_pfn_rejects_non_literal_endings():
    # endings comes from a hub dataset; anything that is not a plain literal
    # must raise instead of being evaluated. literal_eval raises ValueError on
    # non-literal nodes but SyntaxError on malformed input (e.g. a truncated
    # field), so both count as rejection.
    line = {
        "ctx": "sentence",
        "endings": "__import__('os').system('echo pwned')",
        "label": 0,
    }

    with pytest.raises((ValueError, SyntaxError)):
        hellaswag_arabic_pfn(line, "test_task")


def test_hellaswag_arabic_pfn_does_not_execute_endings():
    # pytest.raises alone would also pass if something ran before raising. The payload has to
    # reach its side effect through builtins, because eval() here runs against arabic.py's module
    # globals and cannot see anything defined in this test.
    marker = "lighteval_pwned_by_dataset_field"
    sys.modules.pop(marker, None)
    line = {"ctx": "s", "endings": f"[__import__('sys').modules.setdefault({marker!r}, 1)]", "label": 0}

    try:
        with pytest.raises((ValueError, SyntaxError)):
            hellaswag_arabic_pfn(line, "test_task")
        assert marker not in sys.modules
    finally:
        sys.modules.pop(marker, None)

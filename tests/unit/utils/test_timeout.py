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

import time

import pytest

import lighteval.utils.timeout as timeout_module
from lighteval.metrics.utils.math_comparison import compare_gold_target


@pytest.fixture(params=["native", "windows"])
def make_timeout(request, monkeypatch):
    """Build the timeout decorator for this OS, and also the Windows (non-signal) implementation on any OS."""

    def factory(seconds):
        if request.param == "native":
            return timeout_module.timeout(seconds)
        with monkeypatch.context() as m:
            m.setattr(timeout_module.os, "name", "nt")
            return timeout_module.timeout(seconds)

    return factory


def test_returns_result(make_timeout):
    assert make_timeout(5)(lambda a, b: a + b)(1, 2) == 3


def test_works_with_local_functions(make_timeout):
    # Regression test for https://github.com/huggingface/lighteval/issues/1108: lighteval decorates closures
    # (e.g. compare_gold_target), which the old Windows implementation could not pickle into a child process.
    offset = 10

    def add_offset(x):
        return x + offset

    assert make_timeout(5)(add_offset)(1) == 11


def test_mutations_are_visible_to_the_caller(make_timeout):
    # DynamicMetric.add_to_specifics_with_timeout fills in a field on the object it is given.
    class Doc:
        specific = None

    def set_specific(doc):
        doc.specific = {"extracted": ["42"]}

    doc = Doc()
    make_timeout(5)(set_specific)(doc)
    assert doc.specific == {"extracted": ["42"]}


def test_propagates_exceptions(make_timeout):
    def fail():
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        make_timeout(5)(fail)()


def test_raises_timeout_error(make_timeout):
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        make_timeout(1)(time.sleep)(5)
    assert time.monotonic() - start < 4


def test_math_comparison_matches_identical_answers():
    # On Windows every comparison used to fail inside the timeout wrapper and was scored as wrong.
    assert compare_gold_target(["7"], ["7"])
    assert not compare_gold_target(["7"], ["8"])

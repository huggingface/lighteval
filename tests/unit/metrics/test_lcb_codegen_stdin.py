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

from lighteval.tasks.tasks.lcb.codegen_metrics import call_method


def test_call_method_supports_binary_stdin_buffer():
    """A LiveCodeBench solution may read raw bytes via ``sys.stdin.buffer`` (a
    common fast-I/O idiom). The stdin patch is a ``StringIO``, which has no
    ``.buffer``; without the additive ``.buffer`` patch such a correct solution
    raises ``AttributeError`` and is scored as failing."""

    def solution():
        return sys.stdin.buffer.read()

    assert call_method(solution, "hello\nworld\n") == b"hello\nworld\n"


def test_call_method_text_stdin_unchanged():
    """The existing text path (``sys.stdin.read``) keeps returning the inputs
    unchanged after the additive ``.buffer`` patch."""

    def solution():
        return sys.stdin.read()

    assert call_method(solution, "hello\nworld\n") == "hello\nworld\n"

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


def test_call_method_supports_stdin_buffer():
    """Regression test for #1255: a solution that reads ``sys.stdin.buffer`` must work.

    The grader patches ``sys.stdin`` with a ``StringIO``, which has no ``buffer`` attribute, so
    code doing ``sys.stdin.buffer.read()`` previously failed with an ``AttributeError``.
    """
    captured = {}

    def method():
        captured["data"] = sys.stdin.buffer.read()

    call_method(method, "hello world")

    assert captured["data"] == b"hello world"

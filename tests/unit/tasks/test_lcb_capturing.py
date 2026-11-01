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

"""Tests that `Capturing` accepts every standard way of writing to stdout.

The LCB grader reads a solution's answer out of `Capturing`. The fake stdout it
installs has to expose the same surface as the real one, otherwise a correct
solution is scored on how it writes its answer rather than on the answer.
"""

import sys

import pytest

from lighteval.tasks.tasks.lcb.codegen_metrics import Capturing


EXPECTED = "6"

# Every one of these prints the same answer and must be captured identically.
WRITE_STYLES = {
    "print": "print(6)",
    "stdout_write": "import sys\nsys.stdout.write('6\\n')",
    "stdout_writelines": "import sys\nsys.stdout.writelines(['6', '\\n'])",
    "stdout_buffer_write": "import sys\nsys.stdout.buffer.write(b'6\\n')",
    "stdout_buffer_writelines": "import sys\nsys.stdout.buffer.writelines([b'6', b'\\n'])",
}


def run_solution(source: str) -> str:
    with Capturing() as captured:
        exec(source, {})
    return "".join(captured).strip()


@pytest.mark.parametrize("source", WRITE_STYLES.values(), ids=WRITE_STYLES.keys())
def test_every_way_of_writing_stdout_is_captured(source: str):
    assert run_solution(source) == EXPECTED


def test_text_and_binary_writes_interleave_in_order():
    """A solution mixing print() and buffer.write() keeps its output order."""
    source = "import sys\nprint('a')\nsys.stdout.buffer.write(b'b\\n')\nprint('c')"
    assert run_solution(source) == "a\nb\nc"


def test_buffer_reports_bytes_written():
    """`write` returns the byte count, as a real binary stream does."""
    with Capturing():
        written = sys.stdout.buffer.write(b"abcd")
    assert written == 4


def test_capturing_restores_the_real_stdout():
    """Using the fake buffer must not leave the real stdout replaced."""
    real_stdout = sys.stdout
    run_solution(WRITE_STYLES["stdout_buffer_write"])
    assert sys.stdout is real_stdout
    assert hasattr(sys.stdout, "buffer")


def test_undecodable_bytes_do_not_crash_the_grader():
    """Invalid UTF-8 is replaced rather than raising out of the sandbox."""
    source = "import sys\nsys.stdout.buffer.write(b'\\xff')"
    assert run_solution(source) == "�"

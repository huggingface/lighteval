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

from lighteval.tasks.tasks.lcb.codegen_metrics import Capturing, grade_stdio


@pytest.mark.parametrize(
    "source",
    [
        "print(6)",
        "import sys\nsys.stdout.write('6\\n')",
        "import sys\nsys.stdout.buffer.write(b'6\\n')",
    ],
    ids=["print", "stdout.write", "stdout.buffer.write"],
)
def test_capturing_reads_every_output_idiom(source):
    with Capturing() as captured:
        exec(source, {})

    assert "".join(captured).strip() == "6"


def test_capturing_preserves_order_of_text_and_binary_writes():
    script = (
        "import sys\n"
        "sys.stdout.write('text\\n')\n"
        "sys.stdout.buffer.write(b'buffer\\n')\n"
        "sys.stdout.write('text again\\n')\n"
        "sys.stdout.buffer.writelines([b'lines\\n', b'end\\n'])\n"
    )

    with Capturing() as captured:
        exec(script, {})

    assert "".join(captured) == "text\nbuffer\ntext again\nlines\nend\n"


def test_capturing_buffer_write_returns_byte_count():
    with Capturing() as captured:
        written = sys.stdout.buffer.write(b"abc")

    assert written == 3
    assert captured == ["abc"]


def test_capturing_buffer_replaces_undecodable_bytes():
    with Capturing() as captured:
        sys.stdout.buffer.write(b"\xff\n")

    assert captured == ["\ufffd\n"]


def test_capturing_restores_original_stdout():
    original_stdout = sys.stdout

    with Capturing():
        assert sys.stdout is not original_stdout

    assert sys.stdout is original_stdout


def test_grade_stdio_scores_binary_output_as_correct():
    results = grade_stdio(
        code="import sys\nsys.stdout.buffer.write(b'6\\n')",
        all_inputs=[""],
        all_outputs=["6"],
        timeout=6,
    )

    assert results == [True]

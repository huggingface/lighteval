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

import json
import multiprocessing

import pytest

from lighteval.tasks.tasks.lcb import codegen_metrics


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_check_correctness_without_fork(monkeypatch, start_method):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is not available on this platform")
    ctx = multiprocessing.get_context(start_method)
    monkeypatch.setattr(codegen_metrics.multiprocessing, "Process", ctx.Process)
    monkeypatch.setattr(codegen_metrics.multiprocessing, "Manager", ctx.Manager)

    sample = {"input_output": json.dumps({"inputs": ["1 2\n", "5 7\n"], "outputs": ["3\n", "12\n"]})}
    code = "a, b = map(int, input().split())\nprint(a + b)\n"

    assert codegen_metrics.check_correctness(sample, code, timeout=6) == [True, True]

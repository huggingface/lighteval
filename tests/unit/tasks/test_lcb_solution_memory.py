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
import sys

import pytest

from lighteval.tasks.tasks.lcb import codegen_metrics


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs RLIMIT_AS and /proc")

SAMPLE = {"input_output": json.dumps({"inputs": ["3\n", "4\n"], "outputs": ["3\n", "4\n"], "fn_name": None})}

# Asks for 512 MiB once and then prints the right answer: without a limit it passes.  The sizes are
# kept small because bytearray touches every page, so the run needs that much free memory.
GREEDY = "b = bytearray(512 * 1024 * 1024)\nprint(input())\n"
MODEST = "b = bytearray(32 * 1024 * 1024)\nprint(input())\n"
THREADS = (
    "import threading\n"
    "out = []\n"
    "t = threading.Thread(target=lambda: out.append(input()))\n"
    "t.start()\n"
    "t.join()\n"
    "print(out[0])\n"
)


def pass_at_1(code):
    # The path the metric takes: a process pool, then a forked child per solution.
    metrics, _ = codegen_metrics.codegen_metrics([SAMPLE], [[code]], k_list=[1], num_process_evaluate=1)
    return metrics["pass@1"]


def test_solution_that_allocates_past_the_limit_fails(monkeypatch):
    monkeypatch.setenv(codegen_metrics.SOLUTION_MEMORY_LIMIT_ENV, "128")
    assert pass_at_1(GREEDY) == 0.0


def test_solutions_within_the_limit_still_pass(monkeypatch):
    monkeypatch.setenv(codegen_metrics.SOLUTION_MEMORY_LIMIT_ENV, "128")
    assert pass_at_1(MODEST) == 1.0
    assert pass_at_1(THREADS) == 1.0


@pytest.mark.parametrize("value", ["0", "-1"])
def test_limit_can_be_turned_off(monkeypatch, value):
    monkeypatch.setenv(codegen_metrics.SOLUTION_MEMORY_LIMIT_ENV, value)
    assert pass_at_1(GREEDY) == 1.0


def test_default_limit_leaves_ordinary_solutions_alone(monkeypatch):
    monkeypatch.delenv(codegen_metrics.SOLUTION_MEMORY_LIMIT_ENV, raising=False)
    assert pass_at_1(MODEST) == 1.0

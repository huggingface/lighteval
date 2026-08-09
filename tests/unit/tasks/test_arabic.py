# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from lighteval.tasks.multilingual.tasks.arabic import hellaswag_arabic_pfn


def test_hellaswag_arabic_pfn_parses_literal_endings():
    line = {
        "ctx": "سياق",
        "endings": "['نهاية أولى', 'نهاية ثانية']",
        "label": 1,
    }

    doc = hellaswag_arabic_pfn(line, task_name="test_task")

    assert doc.task_name == "test_task"
    assert doc.choices == ["نهاية أولى", "نهاية ثانية"]
    assert doc.gold_index == 1


def test_hellaswag_arabic_pfn_rejects_non_literal_endings():
    line = {
        "ctx": "سياق",
        "endings": "['safe'] + ['still an expression']",
        "label": 0,
    }

    with pytest.raises(ValueError):
        hellaswag_arabic_pfn(line)

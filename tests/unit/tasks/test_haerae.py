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

from lighteval.tasks.tasks.haerae import (
    HAERAE_CF_TASKS,
    HAERAE_SUBSETS,
    HAERAE_TASKS,
    haerae_cf_prompt,
    haerae_prompt,
)


QUERY = (
    "다음 질문에 대한 정답을 선택지에서 고르시오.\n"
    "### 질문: 우리나라의 수도는 어디인가?\n"
    "### 선택지:\n(A) 부산\n(B) 서울\n(C) 대구\n(D) 인천\n(E) 광주\n"
    "### 정답:"
)


def make_line(answer: str) -> dict:
    return {
        "query": QUERY,
        "options": str(["부산", "서울", "대구", "인천", "광주"]),
        "answer": answer,
    }


@pytest.mark.parametrize("answer, index", [("(A)", 0), ("(B)", 1), ("(C)", 2), ("(D)", 3), ("(E)", 4)])
def test_gold_index(answer, index):
    assert haerae_prompt(make_line(answer), "haerae:history").gold_index == index
    assert haerae_cf_prompt(make_line(answer), "haerae_cf:history").gold_index == index


def test_label_prompt():
    doc = haerae_prompt(make_line("(B)"), "haerae:history")
    assert doc.query == QUERY
    assert doc.choices == [" (A)", " (B)", " (C)", " (D)", " (E)"]


def test_closed_form_prompt():
    doc = haerae_cf_prompt(make_line("(B)"), "haerae_cf:history")
    assert doc.query == "질문: 우리나라의 수도는 어디인가?\n정답:"
    assert doc.choices == [" 부산", " 서울", " 대구", " 인천", " 광주"]
    assert doc.choices[doc.gold_index] == " 서울"


def test_task_table():
    assert len(HAERAE_TASKS) == len(HAERAE_CF_TASKS) == len(HAERAE_SUBSETS) == 5
    assert all(task.hf_revision for task in HAERAE_TASKS + HAERAE_CF_TASKS)

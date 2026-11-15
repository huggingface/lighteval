# MIT License

# Copyright (c) 2026 The HuggingFace Team

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

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.kormedmcqa import (
    SUBSETS,
    TASKS_TABLE,
    extract_choice,
    kormedmcqa_metric,
    kormedmcqa_prompt,
)


@pytest.fixture
def row():
    # Synthetic row with the same columns as sean0042/KorMedMCQA; no dataset text is redistributed.
    return {
        "question": "  다음 중 옳은 것은?  ",
        "A": "보기 하나",
        "B": "보기 둘",
        "C": "보기 셋",
        "D": "보기 넷",
        "E": "보기 다섯",
        "answer": 3,
    }


# ---------- prompt ----------


def test_prompt_exact_snapshot(row):
    doc = kormedmcqa_prompt(row, "kormedmcqa:doctor")
    assert doc.query == ("다음 중 옳은 것은?\nA. 보기 하나\nB. 보기 둘\nC. 보기 셋\nD. 보기 넷\nE. 보기 다섯\n정답：")
    assert doc.choices == ["A", "B", "C", "D", "E"]
    assert doc.gold_index == 2
    assert doc.task_name == "kormedmcqa:doctor"


# Full-width colon cases
def test_prompt_uses_fullwidth_colon(row):
    query = kormedmcqa_prompt(row, "kormedmcqa:doctor").query
    assert query.endswith("정답：")
    assert not query.endswith("정답:")


@pytest.mark.parametrize("answer,letter", [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")])
def test_prompt_gold_is_one_based_answer(row, answer, letter):
    row["answer"] = answer
    doc = kormedmcqa_prompt(row, "kormedmcqa:doctor")
    assert doc.get_golds() == [letter]


# ---------- extract_choice ----------


@pytest.mark.parametrize(
    "output,expected",
    [
        # pattern 1: 정답[:\s]*X
        ("정답: B", "B"),
        ("정답 B", "B"),
        ("정답B", "B"),
        ("정답: b", "B"),  # IGNORECASE, then upper() !
        # pattern 2: 정답은 X입니다
        ("정답은 E입니다", "E"),
        # pattern 3: X.
        ("A. 폐렴", "A"),
        # pattern 4: standalone letter
        ("C", "C"),
        (" D ", "D"),
        # no match
        ("모르겠다", ""),
        ("", ""),
    ],
)
def test_extract_choice_each_pattern(output, expected):
    assert extract_choice(output) == expected


# Check IF one regex answered several answers and whether the last one [-1] is received or not
def test_extract_choice_takes_last_match_of_pattern():
    assert extract_choice("A 아니면 B") == "B"
    assert extract_choice("정답: A, 다시 보니 정답: D") == "D"


def test_extract_choice_earlier_pattern_wins_over_position():
    # Pattern 3 would give "B" and pattern 4 would give "B" (last), but pattern 1 is tried first.
    assert extract_choice("정답: D 이유는 B. 때문") == "D"


def test_extract_choice_fullwidth_colon_is_not_pattern_1():
    # 정답[:\s]* only accepts the ASCII colon, so "정답：C" falls through to pattern 4.
    assert extract_choice("정답：C") == "C"
    # With an ASCII colon pattern 1 keeps the first answer; with a full-width colon pattern 4 takes the last letter.
    assert extract_choice("정답: C 또는 A") == "C"
    assert extract_choice("정답：C 또는 A") == "A"


def test_extract_choice_keeps_reference_regex_gaps():
    # Hangul counts as a word character for \b, so a letter glued to Hangul is not extracted.
    # Kept on purpose to match the lm-evaluation-harness regexes.
    assert extract_choice("정답은 C번") == ""
    assert extract_choice("C번입니다") == ""


# ---------- metric ----------


@pytest.mark.parametrize(
    "generated,score",
    [
        ("C", 1),
        (" C", 1),
        ("정답은 C입니다", 1),
        ("정답: c", 1),
        ("B", 0),
        ("", 0),
        ("모르겠다", 0),
    ],
)
def test_metric_scores_extracted_letter(row, generated, score):
    doc = kormedmcqa_prompt(row, "kormedmcqa:doctor")  # gold is "C"
    response = ModelResponse(text=[generated])
    assert kormedmcqa_metric.sample_level_fn.compute(doc=doc, model_response=response) == score


# ---------- task configs ----------


def test_tasks_table_has_one_task_per_subset():
    assert [task.name for task in TASKS_TABLE] == [f"kormedmcqa:{subset}" for subset in SUBSETS]
    assert [task.hf_subset for task in TASKS_TABLE] == ["doctor", "nurse", "pharm", "dentist"]


@pytest.mark.parametrize("task", TASKS_TABLE, ids=lambda task: task.name)
def test_task_config_matches_harness_settings(task):
    assert task.hf_repo == "sean0042/KorMedMCQA"
    assert task.hf_revision == "79efd6f91edfc8036330d7a4daa88b9f2deb9a82"
    assert task.evaluation_splits == ("test",)
    assert task.few_shots_split == "fewshot"
    assert task.few_shots_select == "sequential"
    assert task.generation_size == 1024
    assert list(task.stop_sequence) == ["Q:", "</s>", "<|im_end|>", ".", "\n\n"]
    assert [metric.metric_name for metric in task.metrics] == ["em"]

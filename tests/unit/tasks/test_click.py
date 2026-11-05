# MIT License

# Copyright (c) 2026 The HuggingFace Team

import copy
import hashlib
import json

import pytest

from lighteval.tasks.tasks.click import CATEGORY_FILES, build_click_payload, click_prompt, raw_row_sha256


@pytest.fixture
def row():
    # All fixtures in this module are synthetic; no CLIcK text is redistributed.
    return {
        "id": "synthetic-not-a-category",
        "paragraph": "",
        "question": "대한민국의 수도는 어디인가?",
        "choices": ["부산", "대구", "서울", "대전"],
        "answer": "서울",
    }


def test_four_options_exact_snapshot(row):
    doc = click_prompt(row, "click:society")
    assert doc.query == (
        "주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n"
        "[질문]\n대한민국의 수도는 어디인가?\n\n"
        "[선택지]\nA. 부산\nB. 대구\nC. 서울\nD. 대전\n\n정답:"
    )
    assert doc.choices == [" A. 부산", " B. 대구", " C. 서울", " D. 대전"]
    assert doc.gold_index == 2
    assert doc.task_name == "click:society"
    assert not doc.instruction
    assert doc.specific["original_id"] == row["id"]
    assert row["id"] not in doc.query


def test_five_options_exact_snapshot(row):
    row.update(paragraph="도시를 골라 보세요.\n지문 두 번째 줄.", choices=["부산", "대구", "인천", "대전", "서울"])
    doc = click_prompt(row, "click:geography")
    assert doc.query == (
        "주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D, E 중에 골라 알파벳 하나로 답하시오.\n\n"
        "[지문]\n도시를 골라 보세요.\n지문 두 번째 줄.\n\n"
        "[질문]\n대한민국의 수도는 어디인가?\n\n"
        "[선택지]\nA. 부산\nB. 대구\nC. 인천\nD. 대전\nE. 서울\n\n정답:"
    )
    assert doc.choices == [" A. 부산", " B. 대구", " C. 인천", " D. 대전", " E. 서울"]
    assert doc.gold_index == 4
    assert (doc.query + doc.choices[4]).endswith("정답: E. 서울")


@pytest.mark.parametrize("count,index", [(n, i) for n in (4, 5) for i in range(n)])
def test_every_gold_position_ignores_id(row, count, index):
    options = ["하나", "둘", "셋", "넷", "다섯"][:count]
    row.update(id="Grammar_CSAT_4_5", choices=options, answer=options[index])
    payload = build_click_payload(row)
    assert payload["gold_index"] == index
    assert len(payload["choices"]) == count
    assert payload["choices"][index] == f" {'ABCDE'[index]}. {options[index]}"


def test_preserve_unicode_spaces_internal_options_and_newlines(row):
    row.update(
        paragraph=" \t漢字·한글\n끝  ",
        question="  질문\nㄱ. 내부 보기\nㄴ. 다음 보기\t",
        choices=["  한글  ", "漢字", "가\n나", "가"],
        answer="가\n나",
    )
    before = copy.deepcopy(row)
    doc = click_prompt(row, "click:grammar")
    assert doc.query == (
        "주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n"
        "[지문]\n \t漢字·한글\n끝  \n\n"
        "[질문]\n  질문\nㄱ. 내부 보기\nㄴ. 다음 보기\t\n\n"
        "[선택지]\nA.   한글  \nB. 漢字\nC. 가\n나\nD. 가\n\n정답:"
    )
    assert doc.choices == [" A.   한글  ", " B. 漢字", " C. 가\n나", " D. 가"]
    assert doc.gold_index == 2
    assert row == before


@pytest.mark.parametrize("paragraph", ["", " ", "\n\t\r", "\u3000"])
def test_blank_paragraph_is_omitted(row, paragraph):
    row["paragraph"] = paragraph
    assert "[지문]" not in build_click_payload(row)["query"]


def test_answer_and_id_do_not_leak_into_model_input(row):
    original = build_click_payload(row)
    row["answer"] = "부산"
    changed = build_click_payload(row)
    assert changed == {**original, "gold_index": 0}
    row["id"] = "unrelated-id"
    assert build_click_payload(row) == changed


@pytest.mark.parametrize("field", ["id", "paragraph", "question", "choices", "answer"])
def test_missing_fields_raise(row, field):
    del row[field]
    with pytest.raises(ValueError):
        click_prompt(row, "click:society")


@pytest.mark.parametrize("field", ["id", "paragraph", "question", "answer"])
@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_string_fields_reject_other_types(row, field, value):
    row[field] = value
    with pytest.raises(ValueError, match=f"{field} must be a string"):
        build_click_payload(row)


@pytest.mark.parametrize(
    "choices",
    [
        None,
        "abcd",
        ("부산", "대구", "서울", "대전"),
        [],
        ["서울"] * 3,
        ["서울"] * 6,
        ["부산", "", "서울", "대전"],
        ["부산", " \n", "서울", "대전"],
        ["부산", None, "서울", "대전"],
        ["부산", 3, "서울", "대전"],
        ["부산", "대구", "대전", "인천"],
        ["서울", "서울", "부산", "대구"],
    ],
)
def test_invalid_options_raise(row, choices):
    row["choices"] = choices
    with pytest.raises(ValueError):
        build_click_payload(row)


@pytest.mark.parametrize("question", ["", " \n\t"])
def test_blank_question_raises(row, question):
    row["question"] = question
    with pytest.raises(ValueError, match="question must be nonempty"):
        build_click_payload(row)


@pytest.mark.parametrize("bad_row", [None, [], "not a row", 0])
def test_non_mapping_raises(bad_row):
    with pytest.raises(ValueError, match="row must be a mapping"):
        build_click_payload(bad_row)


@pytest.mark.parametrize("answer", [" 서울", "서울 ", "서울"])
def test_answer_is_not_normalized(row, answer):
    row["answer"] = answer
    with pytest.raises(ValueError, match="exactly one"):
        build_click_payload(row)


def test_duplicate_ids_and_distractors_are_not_removed(row):
    other = {**row, "choices": ["부산", "부산", "서울", "대전"]}
    first, second = (click_prompt(r, "click:society") for r in (row, other))
    assert first.specific["original_id"] == second.specific["original_id"]
    assert first.specific["raw_row_sha256"] != second.specific["raw_row_sha256"]
    assert second.choices == [" A. 부산", " B. 부산", " C. 서울", " D. 대전"]
    assert second.gold_index == 2


def test_hash_matches_fixed_serialization_and_ignores_extra_metadata(row):
    canonical = (
        '{"answer":"서울","choices":["부산","대구","서울","대전"],'
        '"id":"synthetic-not-a-category","paragraph":"","question":"대한민국의 수도는 어디인가?"}'
    )
    assert raw_row_sha256(row) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert raw_row_sha256({**row, "source_path": "elsewhere"}) == raw_row_sha256(row)
    assert raw_row_sha256(dict(reversed(list(row.items())))) == raw_row_sha256(row)
    assert raw_row_sha256({**row, "paragraph": " "}) != raw_row_sha256(row)
    assert json.loads(json.dumps(click_prompt(row, "click:society").specific)) == {
        "original_id": row["id"],
        "raw_row_sha256": raw_row_sha256(row),
    }


def test_category_files_are_explicit_sorted_and_nonoverlapping():
    assert set(CATEGORY_FILES) == {
        "society",
        "tradition",
        "history",
        "law",
        "politics",
        "economy",
        "geography",
        "popular",
        "textual",
        "functional",
        "grammar",
    }
    paths = [path for files in CATEGORY_FILES.values() for path in files]
    assert len(paths) == len(set(paths)) == 26
    for files in CATEGORY_FILES.values():
        assert list(files) == sorted(files)
    assert "Dataset/Language/Functional/Functional_CSAT.json" in CATEGORY_FILES["functional"]

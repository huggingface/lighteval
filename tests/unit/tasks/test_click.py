# MIT License

# Copyright (c) 2026 The HuggingFace Team

import copy
import hashlib
import json

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from lighteval.logging.info_loggers import MetricsLogger
from lighteval.metrics.metrics_sample import LoglikelihoodAcc
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.click import (
    CATEGORY_FILES,
    HF_REPO,
    TASKS_TABLE,
    build_click_payload,
    click_prompt,
    raw_row_sha256,
)


# Written out so that changing the pinned revision in click.py fails these tests.
PINNED_REVISION = "d61627859645b5e6edc03fd9f835735d8226fa4e"


@pytest.fixture(scope="module")
def click_registry():
    return Registry(tasks="click|0")


def score_choices(scores, gold_index, token_lengths=None):
    doc = Doc(query="정답:", choices=list("ABCDE"[: len(scores)]), gold_index=gold_index)
    lengths = token_lengths or [1] * len(scores)
    response = ModelResponse(logprobs=scores, output_tokens=[[0] * length for length in lengths])
    return TASKS_TABLE[0].metrics[0].compute_sample(doc=doc, model_response=response)["acc"]


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
        "질문: 대한민국의 수도는 어디인가?\n"
        "보기:\nA: 부산, B: 대구, C: 서울, D: 대전\n정답:"
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
        "주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D, E 중에 골라 알파벳 하나로 답하시오.\n\n"
        "맥락: 도시를 골라 보세요.\n지문 두 번째 줄.\n"
        "질문: 대한민국의 수도는 어디인가?\n"
        "보기:\nA: 부산, B: 대구, C: 인천, D: 대전, E: 서울\n정답:"
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
        "주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n"
        "맥락:  \t漢字·한글\n끝  \n"
        "질문:   질문\nㄱ. 내부 보기\nㄴ. 다음 보기\t\n"
        "보기:\nA:   한글  , B: 漢字, C: 가\n나, D: 가\n정답:"
    )
    assert doc.choices == [" A.   한글  ", " B. 漢字", " C. 가\n나", " D. 가"]
    assert doc.gold_index == 2
    assert row == before


@pytest.mark.parametrize("paragraph", ["", " ", "\n\t\r", "\u3000"])
def test_blank_paragraph_is_omitted(row, paragraph):
    row["paragraph"] = paragraph
    assert "맥락:" not in build_click_payload(row)["query"]


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


@pytest.mark.parametrize("count", [4, 5])
@pytest.mark.parametrize("paragraph", ["", "합성 지문"])
def test_instruction_matches_context_for_each_option_count(row, count, paragraph):
    row.update(paragraph=paragraph, choices=["가", "나", "다", "라", "마"][:count], answer="다")
    query = build_click_payload(row)["query"]
    if paragraph:
        assert query.startswith("주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 ")
        assert "\n맥락: 합성 지문\n" in query
    else:
        assert query.startswith("주어진 질문을 천천히 읽고, 적절한 정답을 ")
        assert "맥락:" not in query
    assert ("D, E 중에" in query) == (count == 5)
    assert (", E: 마" in query) == (count == 5)
    assert all(marker not in query for marker in ("[지문]", "[질문]", "[선택지]"))


def test_metric_compares_raw_logprob_sums_without_length_normalization():
    scores, lengths = [-2.0, -3.0, -8.0, -9.0], [4, 12, 4, 4]
    assert score_choices(scores, gold_index=0, token_lengths=lengths) == 1
    assert score_choices(scores, gold_index=1, token_lengths=lengths) == 0

    # The same scores pick B under token normalization, so this case detects normalization leaking in.
    doc = Doc(query="정답:", choices=list("ABCD"), gold_index=1)
    response = ModelResponse(logprobs=scores, output_tokens=[[0] * length for length in lengths])
    assert LoglikelihoodAcc(logprob_normalization=LogProbTokenNorm()).compute(doc, response) == 1


def test_metric_breaks_exact_ties_toward_the_first_choice():
    assert score_choices([-1.0, -1.0, -2.0, -3.0], gold_index=0) == 1
    assert score_choices([-1.0, -1.0, -2.0, -3.0], gold_index=1) == 0


def test_metric_scores_all_five_choices():
    assert score_choices([-9.0, -8.0, -7.0, -6.0, -1.0], gold_index=4) == 1


def test_aggregation_averages_categories_equally_rather_than_items(click_registry):
    metrics_logger = MetricsLogger()
    outcomes = {"click:society|0": [1, 0, 1], "click:law|0": [0]}
    for task_name, values in outcomes.items():
        for value in values:
            metrics_logger.log(task_name, {"acc": value})

    metrics_logger.aggregate(click_registry.load_tasks(), bootstrap_iters=0)

    # Lighteval reports only accuracies; correct and total counts come from the per-item values.
    society = metrics_logger.metrics_values["click:society|0"]["acc"]
    assert (sum(society), len(society)) == (2, 3)
    aggregated = metrics_logger.metric_aggregated
    assert aggregated["click:society|0"]["acc"] == pytest.approx(2 / 3)
    assert aggregated["click:law|0"]["acc"] == 0
    # Lighteval averages categories equally; the item mean over the same four rows would be 2 / 4.
    assert aggregated["click:_average|0"]["acc"] == pytest.approx(1 / 3)


def test_registry_applies_the_designed_settings_to_all_eleven_tasks(click_registry):
    names = [config.name for config in TASKS_TABLE]
    assert len(names) == len(set(names)) == 11
    assert set(click_registry.task_to_configs) == {f"click:{category}" for category in CATEGORY_FILES}

    for category, files in CATEGORY_FILES.items():
        configs = click_registry.task_to_configs[f"click:{category}"]
        assert len(configs) == 1
        config = configs[0]
        assert config.full_name == f"click:{category}|0"
        assert config.prompt_function is click_prompt
        assert (config.hf_repo, config.hf_subset, config.hf_revision) == (HF_REPO, "default", PINNED_REVISION)
        assert config.hf_data_files == {"train": list(files)}
        assert list(config.hf_avail_splits) == list(config.evaluation_splits) == ["train"]
        assert config.few_shots_split is None
        assert config.few_shots_select is None
        assert config.num_fewshots == 0
        assert config.generation_size is None
        assert not config.stop_sequence
        assert config.must_remove_duplicate_docs is False

        assert len(config.metrics) == 1
        metric = config.metrics[0]
        assert metric.metric_name == "acc"
        assert metric.category == SamplingMethod.LOGPROBS
        assert metric.corpus_level_fn is np.mean
        assert type(metric.sample_level_fn) is LoglikelihoodAcc
        assert metric.sample_level_fn.logprob_normalization is None


def test_registered_task_loads_pinned_files_into_zero_shot_logprob_docs(click_registry, monkeypatch, row):
    calls = []

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return DatasetDict({"train": Dataset.from_list([row])})

    monkeypatch.setattr("lighteval.tasks.lighteval_task.load_dataset", fake_load_dataset)
    task = click_registry.load_tasks()["click:society|0"]

    (doc,) = task.get_docs()

    assert len(calls) == 1
    expected = {
        "path": HF_REPO,
        "name": "default",
        "revision": PINNED_REVISION,
        "data_files": {"train": list(CATEGORY_FILES["society"])},
    }
    assert {key: calls[0].get(key) for key in expected} == expected
    assert doc.task_name == "click:society|0"
    assert doc.sampling_methods == [SamplingMethod.LOGPROBS]
    assert doc.fewshot_samples == []
    assert doc.generation_size is None
    assert not doc.stop_sequences
    assert not doc.instruction

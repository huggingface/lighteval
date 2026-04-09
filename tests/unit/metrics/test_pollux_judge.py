# MIT License
#
# Copyright (c) 2024 The HuggingFace Team

"""Unit tests for POLLUX judge helpers and PolluxLLMJudgeMetric (mocked JudgeLM)."""

from unittest.mock import MagicMock

import pytest

from lighteval.metrics.metrics_sample import PolluxLLMJudgeMetric
from lighteval.metrics.utils.judge_utils import (
    get_judge_prompt_pollux,
    parse_pollux_feedback,
    process_judge_response_pollux,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def test_process_judge_response_pollux_parses_score():
    text = "[FEEDBACK] ok [RESULT] 2.5 [END]"
    assert process_judge_response_pollux(text) == pytest.approx(2.5)
    assert parse_pollux_feedback(text) == "ok"


def test_parse_pollux_feedback_empty_when_missing():
    assert parse_pollux_feedback("[RESULT] 1 [END]") == ""
    assert parse_pollux_feedback("") == ""


def test_process_judge_response_pollux_comma_decimal():
    text = "[RESULT] 1,75 [END]"
    assert process_judge_response_pollux(text) == pytest.approx(1.75)


def test_process_judge_response_pollux_missing_returns_zero():
    assert process_judge_response_pollux("no markers here") == 0.0
    assert process_judge_response_pollux("") == 0.0


def test_get_judge_prompt_pollux_messages_and_reference():
    msgs = get_judge_prompt_pollux(
        question="Q?",
        answer="A",
        options=None,
        gold="ref",
        criteria_name="crit",
        rubrics="0: bad 1: ok",
    )
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    body = msgs[0]["content"]
    assert isinstance(body, str)
    assert "### Задание для оценки:\nQ?" in body
    assert "### Эталонный ответ:\nref" in body
    assert "### Ответ для оценки:\nA" in body
    assert "### Критерий оценки:\ncrit" in body
    assert "### Шкала оценивания по критерию:\n0: bad 1: ok" in body


def test_get_judge_prompt_pollux_omits_reference_when_empty():
    msgs = get_judge_prompt_pollux(
        question="Q",
        answer="A",
        gold="",
        criteria_name="c",
        rubrics="r",
    )
    assert "Эталонный ответ" not in msgs[0]["content"]


def test_pollux_metric_compute_batch_mocked():
    metric = PolluxLLMJudgeMetric(
        criteria_name="accuracy",
        rubrics={0: "no", 1: "yes"},
        judge_model_name="dummy-model",
        judge_backend="openai",
        url="http://localhost:8000/v1",
    )
    mock_scores = [1.0, 0.0]
    metric.judge.evaluate_answer_batch = MagicMock(
        return_value=(mock_scores, [{"role": "user", "content": "p"}], ["raw1", "raw2"])
    )
    docs = [
        Doc(query="q1", choices=[], gold_index=0, task_name="t", specific={"reference_answer": "gold1"}),
        Doc(query="q2", choices=[], gold_index=0, task_name="t", specific=None),
    ]
    responses = [
        ModelResponse(text=["a1"]),
        ModelResponse(text=["a2"]),
    ]
    out = metric.compute(responses, docs)
    assert out == [{"pollux_score": 1.0}, {"pollux_score": 0.0}]
    call_kw = metric.judge.evaluate_answer_batch.call_args
    assert call_kw[0][0] == ["q1", "q2"]
    assert call_kw[0][1] == ["a1", "a2"]
    assert call_kw[0][2] == [None, None]
    assert call_kw[0][3] == ["gold1", None]
    assert call_kw[1]["criteria_name"] == ["accuracy", "accuracy"]
    assert call_kw[1]["rubrics"] == ["0: no\n1: yes", "0: no\n1: yes"]


def test_pollux_metric_include_feedback_from_raw():
    metric = PolluxLLMJudgeMetric(
        criteria_name="c",
        rubrics={0: "r"},
        judge_model_name="m",
        judge_backend="openai",
        url="http://localhost:8000/v1",
        include_feedback=True,
    )
    raw_a = "[FEEDBACK] first [RESULT] 1.0 [END]"
    raw_b = "[RESULT] 0.0 [END]"
    metric.judge.evaluate_answer_batch = MagicMock(
        return_value=([1.0, 0.0], [{"role": "user", "content": "p"}], [raw_a, raw_b])
    )
    docs = [
        Doc(query="q1", choices=[], gold_index=0, task_name="t"),
        Doc(query="q2", choices=[], gold_index=0, task_name="t"),
    ]
    responses = [ModelResponse(text=["a1"]), ModelResponse(text=["a2"])]
    out = metric.compute(responses, docs)
    assert out[0] == {"pollux_score": 1.0, "pollux_feedback": "first"}
    assert out[1] == {"pollux_score": 0.0, "pollux_feedback": ""}


def test_pollux_metric_accepts_rubrics_dict_and_normalizes():
    metric = PolluxLLMJudgeMetric(
        criteria_name="c",
        rubrics={2: "good", 0: "bad", 1: "ok"},
        judge_model_name="m",
        judge_backend="openai",
        url="http://localhost:8000/v1",
    )
    metric.judge.evaluate_answer_batch = MagicMock(return_value=([1.0], [{"role": "user", "content": "p"}], ["raw"]))
    docs = [Doc(query="q1", choices=[], gold_index=0, task_name="t")]
    responses = [ModelResponse(text=["a1"])]
    _ = metric.compute(responses, docs)
    call_kw = metric.judge.evaluate_answer_batch.call_args
    assert call_kw[1]["rubrics"] == ["0: bad\n1: ok\n2: good"]


def test_pollux_metric_rejects_string_rubrics():
    with pytest.raises(TypeError, match="rubrics must be a mapping score->description"):
        PolluxLLMJudgeMetric(
            criteria_name="c",
            rubrics="0: bad, 1: ok",
            judge_model_name="m",
            judge_backend="openai",
            url="http://localhost:8000/v1",
        )

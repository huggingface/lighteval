import math
from dataclasses import dataclass

import pytest

from lighteval.metrics.metrics_corpus import CorpusLevelTVDMI
from lighteval.metrics.metrics_sample import JudgeLLMTVDMI
from lighteval.metrics.utils.judge_utils import (
    get_judge_prompt_tvdmi,
    process_judge_response_tvdmi,
)


def test_get_judge_prompt_tvdmi_injects_responses():
    question = "Resp A"
    answer = "Resp B"

    messages = get_judge_prompt_tvdmi(question=question, answer=answer, options=None, gold=None)

    # Should be a single chat message
    assert isinstance(messages, list)
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"

    content = msg["content"]
    # Basic structure checks
    assert "Response A:" in content
    assert "Response B:" in content
    assert "Resp A" in content
    assert "Resp B" in content
    # Should mention A/B grading
    assert "A:" in content
    assert "B:" in content


def test_process_judge_response_tvdmi_maps_A_B():
    assert process_judge_response_tvdmi("A") == 1
    assert process_judge_response_tvdmi("B") == 0
    # Robust to case/whitespace
    assert process_judge_response_tvdmi(" a \n") == 1
    assert process_judge_response_tvdmi(" b\t") == 0


def test_process_judge_response_tvdmi_unknown_falls_back_to_0(caplog):
    with caplog.at_level("WARNING"):
        out = process_judge_response_tvdmi("weird")
    assert out == 0
    # Optional: check that we actually logged something
    assert any("TVD-MI judge" in rec.message for rec in caplog.records)


def test_corpus_level_tvdmi_perfect_critic():
    # Always correct on both positive and negative
    items = [
        {"label": 1, "pred": 1},
        {"label": 1, "pred": 1},
        {"label": 0, "pred": 0},
        {"label": 0, "pred": 0},
    ]

    result = CorpusLevelTVDMI()(items)
    assert "tvd_mi" in result
    assert result["tvd_mi"] == pytest.approx(1.0)


def test_corpus_level_tvdmi_random_critic():
    # 50% TPR, 50% TNR → TVD-MI = 0
    items = [
        {"label": 1, "pred": 1},
        {"label": 1, "pred": 0},
        {"label": 0, "pred": 0},
        {"label": 0, "pred": 1},
    ]

    result = CorpusLevelTVDMI()(items)
    assert result["tvd_mi"] == pytest.approx(0.0)


def test_corpus_level_tvdmi_missing_class_returns_nan():
    # No negatives → TVD-MI undefined
    items = [
        {"label": 1, "pred": 1},
        {"label": 1, "pred": 0},
    ]

    result = CorpusLevelTVDMI()(items)
    assert math.isnan(result["tvd_mi"])


@dataclass
class FakeDoc:
    response_a: str
    response_b: str
    pair_label: int


def test_judge_tvdmi_compute(monkeypatch):
    judge = JudgeLLMTVDMI()

    # Two examples: one positive, one negative
    docs = [
        FakeDoc("A1", "A2", 1),
        FakeDoc("B1", "B2", 0),
    ]

    # Fake judge backend: we want to check what arguments it receives,
    # and return deterministic scores/prompts/responses.
    def fake_evaluate_answer_batch(questions, answers, options, golds, **kwargs):
        # Input wiring checks
        assert questions == ["A1", "B1"]
        assert answers == ["A2", "B2"]
        assert options == [None, None]
        assert golds == [None, None]

        scores = [1, 0]  # predict SAME for first, DIFFERENT for second
        prompts = ["prompt-0", "prompt-1"]
        responses = ["A", "B"]  # raw judge outputs
        return scores, prompts, responses

    # Attach a fake .judge with our method
    class FakeInnerJudge:
        def evaluate_answer_batch(self, *args, **kwargs):
            return fake_evaluate_answer_batch(*args, **kwargs)

    monkeypatch.setattr(judge, "judge", FakeInnerJudge())

    metrics = judge.compute(responses=[], docs=docs)

    assert len(metrics) == 2

    # Check labels and preds propagated correctly
    assert metrics[0]["label"] == 1
    assert metrics[0]["pred"] == 1
    assert metrics[1]["label"] == 0
    assert metrics[1]["pred"] == 0

    # Check extra fields exist (names match your short_judge_name)
    assert any(k.startswith("user_prompt_") for k in metrics[0].keys())
    assert any(k.startswith("judgement_") for k in metrics[0].keys())

# MIT License

# Copyright (c) 2024 The HuggingFace Team

from dataclasses import asdict, dataclass

from lighteval.models.model_output import ModelResponse
from tests.slow_tests.sample_comparison import compare_sample_details


@dataclass
class DetailSample:
    doc: dict
    metric: dict
    model_response: ModelResponse


def make_logits(logit_b: float, logit_c: float) -> list[list[float]]:
    logits = [0.0] * 40
    logits[33] = logit_b
    logits[34] = logit_c
    return [logits]


def make_current_detail(token_id: int, logit_b: float, logit_c: float, metric: float) -> DetailSample:
    return DetailSample(
        doc={"query": "query", "choices": ["A", "B", "C", "D"]},
        metric={"extractive_match": metric},
        model_response=ModelResponse(
            output_tokens=[[token_id, 151645]],
            logits=make_logits(logit_b, logit_c),
        ),
    )


def make_reference_detail(token_id: int, logit_b: float, logit_c: float, metric: float) -> dict:
    return {
        "doc": {"query": "query", "choices": ["A", "B", "C", "D"]},
        "metric": {"extractive_match": metric},
        "model_response": asdict(
            ModelResponse(
                output_tokens=[[token_id, 151645]],
                logits=make_logits(logit_b, logit_c),
            )
        ),
    }


def test_compare_sample_details_ignores_tied_multiple_choice_predictions():
    current_details = {
        "task": [make_current_detail(token_id=34, logit_b=10.0, logit_c=10.0, metric=1.0)],
    }
    reference_details = {
        "task": [make_reference_detail(token_id=33, logit_b=10.0, logit_c=10.0, metric=0.0)],
    }

    assert compare_sample_details(current_details, reference_details) == {}


def test_compare_sample_details_keeps_non_tied_multiple_choice_predictions():
    current_details = {
        "task": [make_current_detail(token_id=34, logit_b=9.0, logit_c=10.0, metric=1.0)],
    }
    reference_details = {
        "task": [make_reference_detail(token_id=33, logit_b=10.0, logit_c=9.0, metric=0.0)],
    }

    differences = compare_sample_details(current_details, reference_details)

    assert differences["task"][0]["sample_index"] == 0
    assert "output_tokens_difference" in differences["task"][0]
    assert "metric_differences" in differences["task"][0]

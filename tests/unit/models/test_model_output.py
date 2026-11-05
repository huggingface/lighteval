from dataclasses import asdict

from lighteval.models.model_output import ModelResponse


def test_model_response_preserves_usage_metadata_when_sliced():
    response = ModelResponse(
        text=["a", "b"],
        output_tokens=[[1], [2]],
        usage_metadata={
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    )

    sliced = response[1]

    assert sliced.text == ["b"]
    assert sliced.usage_metadata == response.usage_metadata


def test_model_response_usage_metadata_round_trips_through_dict():
    response = ModelResponse(
        text=["answer"],
        usage_metadata={
            "prompt_tokens": 42,
            "prompt_tokens_details": {"cached_tokens": 12},
        },
    )

    restored = ModelResponse(**asdict(response))

    assert restored.usage_metadata == response.usage_metadata

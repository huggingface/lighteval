from dataclasses import dataclass

from lighteval.models.endpoints.inference_providers_model import _usage_metadata_from_response


@dataclass
class UsageDetails:
    cached_tokens: int
    cache_write_tokens: int | None = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: UsageDetails


@dataclass
class Response:
    usage: Usage


def test_usage_metadata_from_response_preserves_nested_cache_counters():
    response = Response(
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=UsageDetails(cached_tokens=75),
        )
    )

    assert _usage_metadata_from_response(response) == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 75},
    }


def test_usage_metadata_from_response_accepts_mapping_responses():
    response = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": None},
        }
    }

    assert _usage_metadata_from_response(response) == {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 40},
    }

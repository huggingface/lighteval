import pytest

from lighteval.metrics.harness_compatibility.drop import DropMetrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def test_a_repeated_span_is_not_an_exact_match_of_one_copy():
    """A gold span that appears twice does not exact-match a single copy."""
    metric = DropMetrics()
    repeated = Doc(
        query="Which name is repeated?",
        choices=["cat, cat"],
        gold_index=0,
        specific={"golds_no_preprocessing": [["cat", "cat"]]},
    )

    one_copy = metric.compute(repeated, ModelResponse(text=["cat"]))
    assert one_copy["em"] == 0
    assert one_copy["f1"] == pytest.approx(0.5)

    both_copies = metric.compute(repeated, ModelResponse(text=["cat", "cat"]))
    assert both_copies["em"] == 1.0
    assert both_copies["f1"] == pytest.approx(1.0)

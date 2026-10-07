import math
from dataclasses import asdict

import pytest

from lighteval.metrics.metrics_corpus import CorpusLevelPerplexityMetric
from lighteval.metrics.sample_preparator import PerplexityPreparator, TargetPerplexityPreparator
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


PREPARATOR_TYPES = (PerplexityPreparator, TargetPerplexityPreparator)


@pytest.mark.parametrize("preparator_type", PREPARATOR_TYPES)
@pytest.mark.parametrize(
    "text",
    [
        "alpha beta",
        " alpha beta",
        "alpha beta ",
        " alpha beta ",
        "\talpha beta\t",
        "\nalpha beta\n",
        "\u00a0alpha beta\u00a0",
        "\u2003alpha beta\u2003",
        "alpha\tbeta",
        "alpha\nbeta",
        "你好 世界",
        " 你好 世界 ",
    ],
)
def test_word_units_count_nonempty_whitespace_delimited_words(preparator_type, text):
    preparator = preparator_type(units_type="words")

    assert preparator.count_units(text) == 2


@pytest.mark.parametrize("preparator_type", PREPARATOR_TYPES)
@pytest.mark.parametrize("text,legacy_count", [("", 1), (" ", 2), ("\t\n", 2), ("\u00a0", 2)])
def test_blank_word_units_preserve_legacy_counts(preparator_type, text, legacy_count):
    preparator = preparator_type(units_type="words")

    assert preparator.count_units(text) == legacy_count


@pytest.mark.parametrize("preparator_type", PREPARATOR_TYPES)
@pytest.mark.parametrize("text", [" alpha beta ", "\talpha beta\n", "你好 世界", "\u2003alpha beta\u2003", "", "\t\n"])
def test_byte_units_count_original_utf8_bytes(preparator_type, text):
    preparator = preparator_type(units_type="bytes")

    assert preparator.count_units(text) == len(text.encode("utf-8"))


@pytest.mark.parametrize("preparator_type", PREPARATOR_TYPES)
def test_repeated_prepare_preserves_reference_and_logprobs(preparator_type):
    preparator = preparator_type(units_type="words")

    for reference in [" alpha beta ", "\talpha beta\n", "alpha beta"]:
        document = Doc(query=reference, original_query=reference, choices=[reference], gold_index=0)
        response = ModelResponse(logprobs=[-1.0, -2.0])
        original_document = asdict(document)
        original_response = asdict(response)

        for _iteration in range(2):
            result = preparator.prepare(document, response)

            assert result.weights == 2
            assert result.logprobs == -3.0
            assert asdict(document) == original_document
            assert asdict(response) == original_response


@pytest.mark.parametrize("preparator_type", PREPARATOR_TYPES)
def test_preparator_to_word_perplexity_uses_word_denominator(preparator_type):
    preparator = preparator_type(units_type="words")
    first_document = Doc(query=" alpha beta ", original_query=" alpha beta ", choices=[" alpha beta "], gold_index=0)
    second_document = Doc(query="\tgamma\n", original_query="\tgamma\n", choices=["\tgamma\n"], gold_index=0)
    first_response = ModelResponse(logprobs=[-1.0, -1.0])
    second_response = ModelResponse(logprobs=[-3.0])

    items = [preparator.prepare(first_document, first_response), preparator.prepare(second_document, second_response)]
    metric = CorpusLevelPerplexityMetric("weighted_perplexity")

    assert [item.weights for item in items] == [2, 1]
    assert metric.compute_corpus(items) == pytest.approx(math.exp(5.0 / 3.0))
    assert metric.compute_corpus(list(reversed(items))) == pytest.approx(math.exp(5.0 / 3.0))
    assert metric.compute_corpus(items + items) == pytest.approx(math.exp(5.0 / 3.0))

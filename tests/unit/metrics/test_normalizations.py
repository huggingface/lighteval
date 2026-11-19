# MIT License

# Copyright (c) 2024 The HuggingFace Team

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

from lighteval.metrics.dynamic_metrics import MultilingualQuasiExactMatchMetric
from lighteval.metrics.normalizations import (
    LogProbCharNorm,
    LogProbPMINorm,
    LogProbTokenNorm,
    get_multilingual_normalizer,
    normalize_log_probs,
    remove_articles,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import Extra, is_package_available
from lighteval.utils.language import Language


def test_char_norm():
    choices_logprob = [10.0, 20.0]
    choices_text = [" hell", "world"]

    result = normalize_log_probs(LogProbCharNorm(ignore_first_space=False), choices_logprob, None, choices_text, None)
    assert result == pytest.approx([2.0, 4.0])

    result = normalize_log_probs(LogProbCharNorm(ignore_first_space=True), choices_logprob, None, choices_text, None)
    assert result == pytest.approx([2.5, 4.0])


def test_token_norm():
    choices_logprob = [10.0, 20.0]
    choices_tokens = [[1, 2, 3], [4, 5]]

    result = normalize_log_probs(LogProbTokenNorm(), choices_logprob, None, None, choices_tokens)
    assert result == pytest.approx([3.333333, 10.0])


def test_pmi_norm():
    choices_logprob = [10.0, 20.0]
    unconditioned_logprob = [5.0, 8.0]

    result = normalize_log_probs(LogProbPMINorm(), choices_logprob, unconditioned_logprob, None, None)
    assert result == pytest.approx([5.0, 12.0])


def test_empty_input():
    empty_logprob = []
    empty_text = []
    empty_tokens = []

    # Test with empty inputs
    assert normalize_log_probs(LogProbCharNorm(), empty_logprob, None, empty_text, None) == []
    assert normalize_log_probs(LogProbTokenNorm(), empty_logprob, None, None, empty_tokens) == []
    assert normalize_log_probs(LogProbPMINorm(), empty_logprob, empty_logprob, None, None) == []


needs_multilingual = pytest.mark.skipif(
    not is_package_available(Extra.MULTILINGUAL), reason="requires lighteval[multilingual]"
)


@pytest.mark.parametrize(
    "language, text, expected_words",
    [
        (Language.ENGLISH, "The Beatles", ["Beatles"]),
        (Language.ENGLISH, "An apple", ["apple"]),
        (Language.FRENCH, "Le chat", ["chat"]),
        (Language.FRENCH, "L'été", ["été"]),
        (Language.GERMAN, "Der Hund", ["Hund"]),
        (Language.SPANISH, "El perro", ["perro"]),
        (Language.ITALIAN, "Il cane", ["cane"]),
        # Lowercase articles were always removed; words that merely start with an article are untouched
        (Language.ENGLISH, "the Beatles", ["Beatles"]),
        (Language.ENGLISH, "Theater", ["Theater"]),
    ],
)
def test_remove_articles_ignores_case(language, text, expected_words):
    assert remove_articles(text, language).split() == expected_words


@needs_multilingual
@pytest.mark.parametrize(
    "language, with_article, without_article",
    [
        (Language.ENGLISH, "The Beatles", "beatles"),
        (Language.ENGLISH, "A dog.", "dog"),
        (Language.FRENCH, "Le chat", "chat"),
        (Language.GERMAN, "Der Hund", "hund"),
        (Language.SPANISH, "El perro", "perro"),
        (Language.ITALIAN, "Il cane", "cane"),
    ],
)
def test_multilingual_normalizer_drops_capitalized_articles(language, with_article, without_article):
    normalizer = get_multilingual_normalizer(language)
    assert normalizer(with_article) == normalizer(without_article) == without_article


@needs_multilingual
def test_multilingual_quasi_exact_match_ignores_capitalized_article():
    doc = Doc(query="Which band recorded Abbey Road?", choices=["Beatles"], gold_index=0)
    metric = MultilingualQuasiExactMatchMetric(Language.ENGLISH, "full").sample_level_fn
    assert metric.compute(doc, ModelResponse(text=["The Beatles"])) == 1
    assert metric.compute(doc, ModelResponse(text=["The Rolling Stones"])) == 0

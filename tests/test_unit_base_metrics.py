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

from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.normalizations import helm_normalizer


class TestBaseMetrics:
    def test_exact_match(self):
        em = ExactMatches(strip_strings=True)

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        assert res == 1

        res = em.compute_one_item(
            " The quick brown fox jumps over the lazy dog\n",
            "\n The quick brown fox jumps over the lazy dog ",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog.",
        )
        assert res == 0

        res = em.compute_one_item("The quick brown fox jumps over the lazy dog", "")
        assert res == 0

        res = em.compute_one_item("", "")
        assert res == 0

    def test_quasi_exact_match(self):
        em = ExactMatches(normalize_gold=helm_normalizer, normalize_pred=helm_normalizer)

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        assert res == 1

        res = em.compute_one_item(
            " The quick brown fox jumps over the lazy dog\n",
            "\n The quick brown fox jumps over the lazy dog ",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog.",
        )
        assert res == 1

        res = em.compute_one_item("the quick brown fox, jumps over lazy dog", "quick brown fox jumps over lazy dog.")
        assert res == 1

        res = em.compute_one_item("The quick brown fox jumps over the lazy dog", "")
        assert res == 0

        res = em.compute_one_item("", "")
        assert res == 0

    def test_prefix_exact_match(self):
        em = ExactMatches(
            strip_strings=True,
            type_exact_match="prefix",
        )

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog. And some other stories.",
        )
        assert res == 1

        res = em.compute_one_item(
            "  The quick brown fox jumps over the lazy dog\n",
            "\n The quick brown fox jumps over the lazy dog",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog.",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over lazy dog. And some other stories.",
        )
        assert res == 0

        res = em.compute_one_item("The quick brown fox jumps over the lazy dog", "")
        assert res == 0

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "Complete mismatch",
        )
        assert res == 0

        res = em.compute_one_item("", "")
        assert res == 0

    def test_prefix_quasi_exact_match(self):
        em = ExactMatches(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            type_exact_match="prefix",
        )
        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog. And some other stories.",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick Brown fox jumps over the lazy dog",
            "the quick brown fox jumps over lazy dog. And some other stories.",
        )
        assert res == 1

        res = em.compute_one_item(
            "  The quick brown fox jumps over the lazy dog\n",
            "\n The quick brown fox jumps over the lazy dog",
        )
        assert res == 1

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog.",
        )
        assert res == 1

        res = em.compute_one_item("The quick brown fox jumps over the lazy dog", "")
        assert res == 0

        res = em.compute_one_item(
            "The quick brown fox jumps over the lazy dog",
            "Complete mismatch",
        )
        assert res == 0

        res = em.compute_one_item("", "")
        assert res == 0

    @pytest.mark.skip(reason="Need to understand what it does.")
    def test_pass_at_k_estimator(self):
        assert False

    @pytest.mark.skip(reason="Using nltk metric function, no need to test.")
    def test_f1_score_quasi(self):
        assert False

    @pytest.mark.skip(reason="Using nltk metric function, no need to test.")
    def test_f1(self):
        assert False

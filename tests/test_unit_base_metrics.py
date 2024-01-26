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

    @pytest.mark.skip(reason="Need tp punderstand what it does.")
    def test_pass_at_k_estimator(self):
        assert False

    @pytest.mark.skip(reason="Using nltk metric function, no need to test.")
    def test_f1_score_quasi(self):
        assert False

    @pytest.mark.skip(reason="Using nltk metric function, no need to test.")
    def test_f1(self):
        assert False

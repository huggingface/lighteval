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

import numpy as np
import pytest

from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
    normalized_multi_choice_prob_metric,
    probability_metric,
)
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.normalizations import LogProbCharNorm, helm_normalizer
from lighteval.tasks.requests import Doc


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

    def test_prob(self):
        # Simple case
        prob_metric = probability_metric()
        result = prob_metric.sample_level_fn(logprobs=np.log([0.7]), target_tokens=None)
        assert result == pytest.approx(0.7)

        # Aggregation function test
        prob_min_metric = probability_metric(aggregation_function=np.min)
        result = prob_min_metric.sample_level_fn(logprobs=np.log([0.7, 0.1]), target_tokens=None)
        assert result == pytest.approx(0.1)

    def test_mc_probability_metric(self):
        mc_prob_metric = normalized_multi_choice_prob_metric()
        result = mc_prob_metric.sample_level_fn(
            gold_ixs=[0],
            choices_logprob=np.log([0.35, 0.1, 0.05]),
            unconditioned_logprob=None,
            choices_tokens=None,
            formatted_doc=Doc(choices=["A", "B", "C"], gold_index=0, query=""),
        )
        assert result == pytest.approx(0.7)

        prob_norm_metric = normalized_multi_choice_prob_metric(normalization=LogProbCharNorm())
        result = prob_norm_metric.sample_level_fn(
            gold_ixs=[1],
            choices_logprob=np.log([0.1**2, 0.35**2, 0.05**3]),
            unconditioned_logprob=None,
            choices_tokens=None,
            formatted_doc=Doc(choices=["AA", "BB", "CCC"], gold_index=1, query=""),
        )
        assert result == pytest.approx(0.7)

    def test_acc(self):
        # Test without normalization
        acc_metric = loglikelihood_acc_metric()
        result = acc_metric.sample_level_fn(
            gold_ixs=[0],
            choices_logprob=np.log([0.7, 0.2, 0.3, 0.4]),
            unconditioned_logprob=None,
            choices_tokens=None,
            formatted_doc=Doc(choices=["A", "B", "C", "D"], gold_index=0, query=""),
        )
        assert result == 1  # The highest logprob (3.0) is at index 3, which is not in gold_ixs

        # Test 0 acc
        result = acc_metric.sample_level_fn(
            gold_ixs=[0],
            choices_logprob=np.log([0.1, 0.2, 0.3, 0.4]),
            unconditioned_logprob=None,
            choices_tokens=None,
            formatted_doc=Doc(choices=["A", "B", "C", "D"], gold_index=0, query=""),
        )
        assert result == 0

        # Test with normalization
        acc_norm_metric = loglikelihood_acc_metric(normalization=LogProbCharNorm())
        result_norm = acc_norm_metric.sample_level_fn(
            gold_ixs=[0],
            choices_logprob=np.log([0.5, 0.6]),
            unconditioned_logprob=None,
            choices_tokens=None,
            formatted_doc=Doc(choices=["ABCDE", "AB"], gold_index=0, query=""),
        )
        assert result_norm == 1  # After normalization, "ABCDE" should have the highest score

        # Test with multiple correct solutions
        result_multi = acc_metric.sample_level_fn(
            gold_ixs=[1, 3],
            choices_logprob=np.log([0.5, 0.6, 0.7, 0.8]),
            unconditioned_logprob=None,
            choices_tokens=None,
            formatted_doc=Doc(choices=["A", "B", "C", "D"], gold_index=[1, 3], query=""),
        )
        assert result_multi == 1

        # Test when the highest logprob is not in gold_ixs
        result_incorrect = acc_metric.sample_level_fn(
            gold_ixs=[1, 2],
            choices_logprob=np.log([0.5, 0.6, 0.7, 0.8]),
            unconditioned_logprob=None,
            choices_tokens=None,
            formatted_doc=Doc(choices=["A", "B", "C", "D"], gold_index=[1, 3], query=""),
        )
        assert result_incorrect == 0

    @pytest.mark.skip(reason="Need to understand what it does.")
    def test_pass_at_k_estimator(self):
        assert False

    @pytest.mark.skip(reason="Using nltk metric function, no need to test.")
    def test_f1_score_quasi(self):
        assert False

    @pytest.mark.skip(reason="Using nltk metric function, no need to test.")
    def test_f1(self):
        assert False

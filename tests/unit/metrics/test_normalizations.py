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

from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm, normalize_log_probs


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


def test_char_norm_requires_one_text_per_logprob():
    choices_logprob = [-1.0, -2.0, -3.0]
    choices_text = ["abc", "de"]
    with pytest.raises(ValueError, match="same length"):
        normalize_log_probs(LogProbCharNorm(ignore_first_space=False), choices_logprob, None, choices_text, None)
    with pytest.raises(ValueError, match="same length"):
        normalize_log_probs(LogProbCharNorm(ignore_first_space=True), choices_logprob, None, choices_text, None)


def test_char_norm_rejects_empty_choice_text():
    with pytest.raises(ValueError, match="non-empty"):
        normalize_log_probs(LogProbCharNorm(), [-1.0], None, [""], None)
    with pytest.raises(ValueError, match="non-empty"):
        normalize_log_probs(LogProbCharNorm(ignore_first_space=True), [-1.0], None, [" "], None)


def test_char_norm_ignores_only_a_leading_space():
    result = normalize_log_probs(LogProbCharNorm(ignore_first_space=True), [-2.0], None, [" ab"], None)
    assert result == pytest.approx([-1.0])

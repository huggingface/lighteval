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

import lighteval.metrics.normalizations as normalizations
from lighteval.metrics.metrics_sample import SamplingMetric


def test_string_normalize_resolves_to_function():
    """A valid normalizer name passed as a string resolves to the function.

    Regression test for the string `normalize` argument being unusable because
    `inspect.getmembers(...)` (a list of tuples) was treated as a dict.
    """
    metric = SamplingMetric(normalize="helm_normalizer")

    assert metric.normalize is normalizations.helm_normalizer
    # It is actually applied during preprocessing.
    assert metric.preprocess("  The  Cat ") == normalizations.helm_normalizer("  The  Cat ")


def test_string_normalize_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown normalization function"):
        SamplingMetric(normalize="not_a_real_normalizer")


def test_callable_normalize_is_kept():
    def custom(text: str) -> str:
        return text.upper()

    metric = SamplingMetric(normalize=custom)

    assert metric.normalize is custom


def test_none_normalize_is_kept():
    metric = SamplingMetric(normalize=None)

    assert metric.normalize is None

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

import functools

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import gsm8k_normalizer, math_normalizer


def test_metric_name_keeps_scalar_sample_params():
    metric = Metrics.exact_match(sample_params={"strip_strings": False})

    assert metric.metric_name == "em:strip_strings=False"


def test_metric_name_uses_function_names_for_callable_sample_params():
    metric = Metrics.exact_match(sample_params={"normalize_gold": math_normalizer, "normalize_pred": gsm8k_normalizer})

    assert metric.metric_name == "em:normalize_gold=math_normalizer&normalize_pred=gsm8k_normalizer"


def test_metric_name_uses_wrapped_function_name_for_partial_sample_params():
    metric = Metrics.exact_match(sample_params={"normalize_gold": functools.partial(math_normalizer)})

    assert metric.metric_name == "em:normalize_gold=partial(math_normalizer, ...)"


def test_metric_name_is_stable_across_instances():
    # Callables must not be rendered through their default repr, as it contains a memory address:
    # it would change at every run, and with it the task hash used for caching
    first = Metrics.exact_match(sample_params={"normalize_gold": math_normalizer})
    second = Metrics.exact_match(sample_params={"normalize_gold": lambda text: text})

    assert first.metric_name == second.metric_name.replace("<lambda>", "math_normalizer")
    assert "0x" not in first.metric_name
    assert "0x" not in second.metric_name


def test_grouping_metric_name_uses_function_names_for_callable_sample_params():
    metric = Metrics.drop(sample_params={"normalize_gold": math_normalizer})

    assert metric.metric_name == ["em:normalize_gold=math_normalizer", "f1:normalize_gold=math_normalizer"]

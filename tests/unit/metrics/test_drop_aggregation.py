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

"""Regression tests for Metrics.drop corpus aggregation (issue #1395)."""

import numpy as np

from lighteval.metrics.harness_compatibility.drop import DropMetrics
from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def test_drop_corpus_aggregation_uses_mean_not_max():
    """One correct document must not report task-level em/f1 of 1.0."""
    aggregations = Metrics.drop.value.get_corpus_aggregations()

    per_sample_em = [1.0, 0.0, 0.0, 0.0]
    per_sample_f1 = [1.0, 0.0, 0.0, 0.0]

    assert aggregations["em"](per_sample_em) == np.mean(per_sample_em)
    assert aggregations["f1"](per_sample_f1) == np.mean(per_sample_f1)
    assert aggregations["em"](per_sample_em) == 0.25
    assert aggregations["f1"](per_sample_f1) == 0.25


def test_drop_sample_level_still_maxes_over_golds():
    """Max over alternate gold answers remains a per-sample behavior."""
    metric = DropMetrics()
    doc = Doc(
        query="q",
        choices=["4"],
        gold_index=0,
        specific={"golds_no_preprocessing": ["4", "four"]},
    )
    response = ModelResponse(text=["4"])

    result = metric.compute(doc=doc, model_response=response)
    assert result["em"] == 1.0
    assert result["f1"] == 1.0

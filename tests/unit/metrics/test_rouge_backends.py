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

import random
import sys

import numpy as np
import pytest

from lighteval.metrics.metrics_sample import ROUGE
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


pytest.importorskip("rouge_score_rs")

WORDS = "the a cats cat running ran relational news dying 2024 U.S. café Summary summaries model".split()


class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()


def make_samples(count=200, seed=0):
    rng = random.Random(seed)

    def text():
        return "".join(rng.choice(WORDS) + rng.choice([" ", ", ", "\n"]) for _ in range(rng.randint(1, 30)))

    samples = []
    for _ in range(count):
        golds = [text() for _ in range(rng.randint(1, 3))]
        doc = Doc(query="", choices=golds, gold_index=list(range(len(golds))))
        samples.append((doc, ModelResponse(text=[text()])))
    return samples


def compute_all(metric_kwargs, samples):
    metric = ROUGE(**metric_kwargs)
    np.random.seed(0)
    return metric, [metric.compute(doc=doc, model_response=response) for doc, response in samples]


@pytest.mark.parametrize(
    "metric_kwargs",
    [
        {"methods": ["rouge1", "rouge2", "rougeL", "rougeLsum"]},
        {"methods": ["rouge1", "rougeLsum"], "multiple_golds": True},
        {"methods": ["rouge1", "rouge2", "rougeL"], "bootstrap": True},
        {"methods": "rougeL", "tokenizer": WhitespaceTokenizer()},
        {"methods": ["rouge1", "rougeL"], "normalize_gold": str.lower, "normalize_pred": str.lower},
    ],
)
def test_rust_backend_matches_rouge_score(metric_kwargs, monkeypatch):
    samples = make_samples()
    fast_metric, fast_scores = compute_all(metric_kwargs, samples)
    assert type(fast_metric.scorer).__module__.startswith("rouge_score_rs")

    monkeypatch.setitem(sys.modules, "rouge_score_rs", None)
    reference_metric, reference_scores = compute_all(metric_kwargs, samples)
    assert type(reference_metric.scorer).__module__.startswith("rouge_score.")

    assert fast_scores == reference_scores

# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random

from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
from lighteval.tasks.requests import Doc


def test_loglikelihood_rolling_handles_empty_choices_without_changing_seed_order(tmp_path):
    model = DummyModel(DummyModelConfig(seed=42, cache_dir=str(tmp_path)))
    model._cache = None
    docs = [
        Doc(task_name="ppl", query="first text", choices=None, gold_index=None),
        Doc(task_name="ppl", query="second text", choices=[], gold_index=None),
        Doc(task_name="ppl", query="multiple choices", choices=["a", "b"], gold_index=None),
    ]

    responses = model.loglikelihood_rolling(docs)

    expected_random = random.Random(42)
    assert [len(response.logprobs) for response in responses] == [1, 1, 2]
    assert [score for response in responses for score in response.logprobs] == [
        -expected_random.random() for _ in range(4)
    ]
    assert [response.argmax_logits_eq_gold for response in responses] == [[False], [False], [False, False]]

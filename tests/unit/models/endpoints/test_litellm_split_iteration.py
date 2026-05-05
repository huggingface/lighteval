# MIT License
#
# Copyright (c) 2026 The HuggingFace Team
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

from types import SimpleNamespace

import pytest

from lighteval.data import GenerativeTaskDataset
from lighteval.models.endpoints.litellm_model import LiteLLMClient
from lighteval.tasks.requests import Doc


pytest.importorskip("litellm")


def _make_doc(query: str, stop_sequences: list[str]) -> Doc:
    return Doc(
        query=query,
        choices=[""],
        gold_index=0,
        generation_size=8,
        stop_sequences=stop_sequences,
        use_logits=False,
        num_samples=1,
    )


def _mock_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=None))],
    )


def test_greedy_until_uses_split_local_contexts():
    docs = [
        _make_doc("alpha", ["A"]),
        _make_doc("beta", ["A"]),
        _make_doc("gamma", ["B"]),
    ]

    expected_dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=LiteLLMClient.DATASET_SPLITS)
    expected_contexts_by_split = [
        [f"ctx:{doc.query}" for doc in split] for split in expected_dataset.splits_iterator()
    ]

    model = LiteLLMClient.__new__(LiteLLMClient)
    model._cache = None
    model.generation_parameters = SimpleNamespace(temperature=1)
    model.prompt_manager = SimpleNamespace(prepare_prompt_api=lambda doc: f"ctx:{doc.query}")

    observed_contexts_by_split: list[list[str]] = []

    def fake_call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence):
        observed_contexts_by_split.append(list(contexts))
        return [_mock_response(f"out-{index}") for index, _ in enumerate(contexts)]

    model._LiteLLMClient__call_api_parallel = fake_call_api_parallel

    results = model.greedy_until(docs)

    assert observed_contexts_by_split == expected_contexts_by_split
    assert sum(len(contexts) for contexts in observed_contexts_by_split) == len(docs)
    assert len(results) == len(docs)

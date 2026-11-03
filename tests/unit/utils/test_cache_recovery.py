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

from dataclasses import asdict

import pytest
from datasets import Dataset, load_dataset

from lighteval.models.dummy.dummy_model import DummyModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached


@pytest.mark.parametrize("sampling_method", list(SamplingMethod))
@pytest.mark.parametrize("legacy_schema", [False, True], ids=["null-response", "obsolete-response-field"])
def test_recomputed_invalid_rows_replace_old_cache(tmp_path, sampling_method, legacy_schema):
    """A rejected persisted row must not win over its successfully recomputed response."""
    config = DummyModelConfig(model_name="cache-recovery", cache_dir=str(tmp_path))
    cache = SampleCache(config)
    task_id = cache.get_task_id("recovery", sampling_method)
    path = cache.get_cache_path(task_id)
    path.parent.mkdir(parents=True)
    preserved = ModelResponse(text=["preserved"], logprobs=[-0.25])
    rows = [
        {"sample_id": "bad", "sample": {"obsolete_response": "old"} if legacy_schema else None},
    ]
    if not legacy_schema:
        rows.append({"sample_id": "good", "sample": asdict(preserved)})
    Dataset.from_list(rows).to_parquet(str(path))

    docs = [Doc(id=id_, task_name="recovery", query=id_, choices=["answer"], gold_index=0) for id_ in ["new", "bad"]]
    if not legacy_schema:
        docs.insert(1, Doc(id="good", task_name="recovery", query="good", choices=["answer"], gold_index=0))

    class Model:
        def __init__(self):
            self._cache = SampleCache(config)
            self.generated = []

        @cached(sampling_method)
        def run(self, docs):
            self.generated.extend(doc.id for doc in docs)
            return [ModelResponse(text=[doc.id], logprobs=[-1.0]) for doc in docs]

    model = Model()
    assert model._cache.existing_indices[task_id] == ([] if legacy_schema else ["good"])
    results = model.run(docs)
    assert model.generated == ["new", "bad"]
    assert [result.text for result in results] == (
        [["new"], ["bad"]] if legacy_schema else [["new"], ["preserved"], ["bad"]]
    )
    if not legacy_schema:
        assert asdict(results[1]) == asdict(preserved)

    persisted = load_dataset("parquet", data_files=str(path), split="train")
    assert len(persisted) == len(docs)
    assert {row["sample_id"] for row in persisted} == {doc.id for doc in docs}
    assert all("obsolete_response" not in row["sample"] for row in persisted)

    reopened = Model()
    assert reopened.run(docs) == results
    assert reopened.generated == []


def test_valid_cached_response_keeps_precedence(tmp_path, caplog):
    """Recovery must not replace existing valid model outputs on accidental reprocessing."""
    cache = SampleCache(DummyModelConfig(model_name="cache-recovery", cache_dir=str(tmp_path)))
    method = SamplingMethod.GENERATIVE
    doc = Doc(id="good", task_name="recovery", query="question", choices=["answer"], gold_index=0)
    task_id = cache.get_task_id(doc.task_name, method)
    old = ModelResponse(text=["original"])
    cache.cache_samples([doc], [old], [task_id], method)
    cache.cache_samples([doc], [ModelResponse(text=["replacement"])], [task_id], method)
    assert cache.get_samples_from_cache([doc], [task_id], method) == [old]
    assert "ignore the new version" in caplog.text

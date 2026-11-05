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


from datasets import Dataset, DatasetDict

from lighteval.models.dummy.dummy_model import DummyModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache


def dummy_prompt_function(item, task_name):
    return Doc(query=item["text"], choices=["A", "B"], gold_index=0, task_name=task_name)


def test_revision_check():
    # Test with a different revision
    cfg_with_revision = LightevalTaskConfig(
        name="test_task_revision",
        prompt_function=dummy_prompt_function,
        hf_repo="lighteval-tests-datasets/dataset-test-1",
        hf_subset="default",
        evaluation_splits=["train"],
        metrics=[],
        hf_revision="25175defadfde48b131b7cd7573ad6f59f868306",
    )
    task_with_revision = LightevalTask(cfg_with_revision)
    docs = task_with_revision.eval_docs()
    queries = [doc.query for doc in docs]
    assert queries == ["hi", "how are you?"]


def test_dataset_filter():
    # Setup

    cfg = LightevalTaskConfig(
        name="test_task",
        prompt_function=dummy_prompt_function,
        hf_repo="lighteval-tests-datasets/dataset-test-1",
        hf_subset="default",
        hf_filter=lambda x: x["text"] == "hi",
        metrics=[],
        evaluation_splits=["train"],
    )
    task = LightevalTask(cfg)

    filtered_docs = task.eval_docs()
    assert len(filtered_docs) == 1
    assert filtered_docs[0].query == "hi"


def test_hf_data_files(tmp_path):
    # create a small jsonl dataset
    data_file = tmp_path / "data.jsonl"
    src_docs = [f"document {i}" for i in range(3)]
    data_file.write_text("\n".join([f'{{"text": "{doc}"}}' for doc in src_docs]))

    cfg = LightevalTaskConfig(
        name="test_data_files",
        prompt_function=dummy_prompt_function,
        hf_repo="json",
        hf_subset="default",
        metrics=[],
        evaluation_splits=["train"],
        hf_data_files=str(data_file),
    )
    task = LightevalTask(cfg)

    eval_docs = task.eval_docs()
    assert [doc.query for doc in eval_docs] == src_docs


def test_doc_ids_are_unique_across_evaluation_splits(tmp_path):
    cfg = LightevalTaskConfig(
        name="multi_split_task",
        prompt_function=dummy_prompt_function,
        hf_repo="unused",
        hf_subset="default",
        metrics=[],
        evaluation_splits=["validation", "test"],
    )
    task = LightevalTask(cfg)
    task.dataset = DatasetDict(
        {
            "validation": Dataset.from_dict({"text": ["validation 0", "validation 1"]}),
            "test": Dataset.from_dict({"text": ["test 0"]}),
        }
    )

    multi_split_docs = task._get_docs_from_split(["validation", "test"])
    single_split_docs = task._get_docs_from_split(["test"])

    assert [doc.id for doc in multi_split_docs] == ["validation:0", "validation:1", "test:0"]
    assert [doc.id for doc in single_split_docs] == ["0"]

    cache = SampleCache(DummyModelConfig(cache_dir=str(tmp_path)))
    sampling_method = SamplingMethod.GENERATIVE
    task_id = cache.get_task_id(multi_split_docs[0].task_name, sampling_method)
    legacy_doc = Doc(
        id="0",
        task_name=multi_split_docs[0].task_name,
        query="legacy",
        choices=["A", "B"],
        gold_index=0,
    )
    cache.cache_samples([legacy_doc], [ModelResponse(text=["legacy"])], [task_id], sampling_method)

    uncached_docs, cached_tasks = cache.get_samples_to_process_and_cache(multi_split_docs, sampling_method)
    assert uncached_docs == multi_split_docs
    assert cached_tasks == set()

    responses = [ModelResponse(text=[doc.query]) for doc in multi_split_docs]
    cache.cache_samples(multi_split_docs, responses, [task_id], sampling_method)
    assert [
        response.text for response in cache.get_samples_from_cache(multi_split_docs, [task_id], sampling_method)
    ] == [[doc.query] for doc in multi_split_docs]

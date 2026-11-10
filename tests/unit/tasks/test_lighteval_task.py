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

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.requests import Doc


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


def _write_id_dataset(tmp_path):
    data_file = tmp_path / "data.jsonl"
    rows = [
        '{"id": "a", "text": "doc a"}',
        '{"id": "b", "text": "doc b"}',
        '{"id": "c", "text": "doc c"}',
    ]
    data_file.write_text("\n".join(rows))
    return data_file


def test_few_shots_id_list_selection(tmp_path):
    # Only the samples whose `id` is in few_shots_id_list should be used as few-shot examples.
    data_file = _write_id_dataset(tmp_path)

    cfg = LightevalTaskConfig(
        name="test_fewshot_ids",
        prompt_function=dummy_prompt_function,
        hf_repo="json",
        hf_subset="default",
        metrics=[],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_id_column="id",
        few_shots_id_list=["a", "c"],
        hf_data_files=str(data_file),
    )
    task = LightevalTask(cfg)

    fewshot_docs = task.fewshot_docs()
    assert [doc.query for doc in fewshot_docs] == ["doc a", "doc c"]

    # Eval docs are unaffected by the few-shot id restriction.
    eval_docs = task.eval_docs()
    assert sorted(doc.query for doc in eval_docs) == ["doc a", "doc b", "doc c"]


def test_few_shots_id_list_without_column_raises():
    with pytest.raises(ValueError):
        LightevalTaskConfig(
            name="test_fewshot_ids_invalid",
            prompt_function=dummy_prompt_function,
            hf_repo="json",
            hf_subset="default",
            metrics=[],
            evaluation_splits=["train"],
            few_shots_id_list=["a"],
        )


def test_few_shots_id_list_missing_column_raises(tmp_path):
    data_file = _write_id_dataset(tmp_path)

    cfg = LightevalTaskConfig(
        name="test_fewshot_ids_missing_col",
        prompt_function=dummy_prompt_function,
        hf_repo="json",
        hf_subset="default",
        metrics=[],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_id_column="does_not_exist",
        few_shots_id_list=["a"],
        hf_data_files=str(data_file),
    )
    task = LightevalTask(cfg)

    with pytest.raises(KeyError):
        task.fewshot_docs()

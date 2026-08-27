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


import logging

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


def _num_samples_config(num_samples, metrics=None):
    return LightevalTaskConfig(
        name="test_num_samples",
        prompt_function=dummy_prompt_function,
        hf_repo="lighteval-tests-datasets/dataset-test-1",
        hf_subset="default",
        evaluation_splits=["train"],
        metrics=metrics or [],
        num_samples=num_samples,
    )


def test_num_samples_defaults_to_one_when_unset():
    cfg = _num_samples_config(num_samples=None)
    task = LightevalTask(cfg)
    assert task.num_samples == [1]


def test_num_samples_config_is_respected():
    # Regression test for the config-level `num_samples` being silently dropped:
    # the request builder takes max(num_samples), so the override must survive.
    cfg = _num_samples_config(num_samples=[16])
    task = LightevalTask(cfg)
    assert max(task.num_samples) == 16
    # 1 stays present so base generative scoring still works.
    assert 1 in task.num_samples


def test_num_samples_config_accepts_bare_int():
    cfg = _num_samples_config(num_samples=8)
    task = LightevalTask(cfg)
    assert max(task.num_samples) == 8


def test_num_samples_without_sampling_metric_warns(caplog):
    cfg = _num_samples_config(num_samples=[16])
    with caplog.at_level(logging.WARNING):
        LightevalTask(cfg)
    assert any("num_samples" in record.message for record in caplog.records)

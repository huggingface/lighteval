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

from lighteval.data import GenerativeTaskDataset
from lighteval.metrics.utils.metric_utils import SamplingMethod
from lighteval.tasks.requests import Doc


# test data that will need to be sorted by length of the string
TEST_DATA = [
    Doc(
        query="1 The quick brown fox jumps over the lazy dog",
        choices=["A", "B", "C"],
        gold_index=0,
    ),
    Doc(
        query="2 The quick brown fox jumps over the lazy dog njsa",
        choices=["A", "B", "C"],
        gold_index=0,
    ),
    Doc(
        query="Some text",
        choices=["A", "B", "C"],
        gold_index=0,
    ),
    Doc(
        query="some more text",
        choices=["A", "B", "C"],
        gold_index=0,
    ),
    Doc(
        query="not sure what to write here",
        choices=["A", "B", "C"],
        gold_index=0,
    ),
]

DATASET_SPLITS = 1


class TestReorderGenerativeTaskDataset:
    def test_reorder_dataset(self):
        data = TEST_DATA.copy()
        for d in data:
            d.task_name = "test"
            d.sampling_methods = [SamplingMethod.GENERATIVE]
            d.generation_size = 10
            d.stop_sequences = ["stop", ":", "end"]

        dataset = GenerativeTaskDataset(requests=data, num_dataset_splits=DATASET_SPLITS)

        sorted_data = dataset.sorted_data
        original_data = dataset.get_original_order(sorted_data)

        for i in range(len(sorted_data) - 1):
            assert len(sorted_data[i].query) >= len(sorted_data[i + 1].query), (
                f"dataset[{i}][0] = {sorted_data[i].query} is shorter than dataset[{i + 1}][0] = {sorted_data[i + 1].query}"
            )

        assert len(sorted_data) == len(original_data), (
            f"reordered dataset has length {len(sorted_data)}, should be {len(dataset)}"
        )

        for sorted_data, orignal in zip(original_data, data):
            assert sorted_data == orignal

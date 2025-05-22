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

from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset
from lighteval.metrics.utils.metric_utils import MetricCategory
from lighteval.tasks.requests import GreedyUntilRequest


# test data that will need to be sorted by length of the string
TEST_DATA = [
    GreedyUntilRequest(
        task_name="test",
        sample_index=0,
        request_index=0,
        context="1 The quick brown fox jumps over the lazy dog",
        stop_sequence=[":", "stop"],
        generation_size=10,
        metric_categories=[MetricCategory.GENERATIVE],
    ),
    GreedyUntilRequest(
        task_name="test",
        sample_index=2,
        request_index=0,
        context="2 The quick brown fox jumps over the lazy dog njsa",
        stop_sequence=[":", "stop"],
        generation_size=10,
        metric_categories=[MetricCategory.GENERATIVE],
    ),
    GreedyUntilRequest(
        task_name="test",
        sample_index=5,
        request_index=0,
        context="Some text",
        stop_sequence=[":", "stop"],
        generation_size=10,
        metric_categories=[MetricCategory.GENERATIVE],
    ),
    GreedyUntilRequest(
        task_name="test",
        sample_index=21,
        request_index=0,
        context="some more text",
        stop_sequence=[":", "stop"],
        generation_size=10,
        metric_categories=[MetricCategory.GENERATIVE],
    ),
    GreedyUntilRequest(
        task_name="test",
        sample_index=1,
        request_index=0,
        context="not sure what to write here",
        stop_sequence=[":", "stop"],
        generation_size=10,
        metric_categories=[MetricCategory.GENERATIVE],
    ),
]

DATASET_SPLITS = 1


class TestReorderGenerativeTaskDataset:
    def test_reorder_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        data = TEST_DATA.copy()
        for request in data:
            request.tokenized_context = tokenizer.encode(request.context)

        dataset = GenerativeTaskDataset(requests=data, num_dataset_splits=DATASET_SPLITS)

        sorted_data = dataset.sorted_data
        original_data = dataset.get_original_order(sorted_data)

        for i in range(len(sorted_data) - 1):
            assert len(sorted_data[i].context) >= len(sorted_data[i + 1].context), (
                f"dataset[{i}][0] = {sorted_data[i].context} is shorter than dataset[{i + 1}][0] = {sorted_data[i + 1].context}"
            )

        assert len(sorted_data) == len(original_data), (
            f"reordered dataset has length {len(sorted_data)}, should be {len(dataset)}"
        )

        for sorted_data, orignal in zip(original_data, data):
            assert sorted_data == orignal

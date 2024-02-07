import pytest
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset
from lighteval.tasks.requests import GreedyUntilRequest


# test data that will need to be sorted by length of the string
TEST_DATA = [
    GreedyUntilRequest(
        task_name="test",
        example_index=0,
        request_index=0,
        context="1 The quick brown fox jumps over the lazy dog",
        stop_sequence=[":", "stop"],
        generation_size=10,
    ),
    GreedyUntilRequest(
        task_name="test",
        example_index=2,
        request_index=0,
        context="2 The quick brown fox jumps over the lazy dog njsa",
        stop_sequence=[":", "stop"],
        generation_size=10,
    ),
    GreedyUntilRequest(
        task_name="test",
        example_index=5,
        request_index=0,
        context="Some text",
        stop_sequence=[":", "stop"],
        generation_size=10,
    ),
    GreedyUntilRequest(
        task_name="test",
        example_index=21,
        request_index=0,
        context="some more text",
        stop_sequence=[":", "stop"],
        generation_size=10,
    ),
    GreedyUntilRequest(
        task_name="test",
        example_index=1,
        request_index=0,
        context="not sure what to write here",
        stop_sequence=[":", "stop"],
        generation_size=10,
    ),
]

DATASET_SPLITS = 1


class TestReorderGenerativeTaskDataset:
    def test_dataset_needs_tokenization(self):
        with pytest.raises(ValueError):
            GenerativeTaskDataset(requests=TEST_DATA, dataset_splits=DATASET_SPLITS)

    def test_reorder_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        data = TEST_DATA.copy()
        for request in data:
            request.tokenized_context = tokenizer.encode(request.context)

        dataset = GenerativeTaskDataset(requests=data, dataset_splits=DATASET_SPLITS)

        sorted_data = dataset.sorted_data
        original_data = dataset.get_original_order(sorted_data)

        for i in range(len(sorted_data) - 1):
            assert (
                len(sorted_data[i].context) >= len(sorted_data[i + 1].context)
            ), f"dataset[{i}][0] = {sorted_data[i].context} is shorter than dataset[{i+1}][0] = {sorted_data[i+1].context}"

        assert len(sorted_data) == len(
            original_data
        ), f"reordered dataset has length {len(sorted_data)}, should be {len(dataset)}"

        for sorted_data, orignal in zip(original_data, data):
            assert sorted_data == orignal

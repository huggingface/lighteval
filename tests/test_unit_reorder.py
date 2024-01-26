from lighteval.data import GenerativeTaskDataset


# test data that will need to be sorted by length of the string
data = [
    ("1 The quick brown fox jumps over the lazy dog", ([":", "stop"], 10)),
    ("2 The quick brown fox jumps over the lazy dog njsa", ([":", "stop"], 10)),
    ("Some text", ([":", "stop"], 10)),
    ("some more text", ([":", "stop"], 10)),
    ("not sure what to write here", ([":", "stop"], 10)),
]

DATASET_SPLITS = 1


class TestReorderGenerativeTaskDataset:
    def test_reorder_dataset(self):
        dataset = GenerativeTaskDataset(requests=data, dataset_splits=DATASET_SPLITS)

        sorted_data = dataset.sorted_data
        original_data = dataset.get_original_order(sorted_data)

        for i in range(len(sorted_data) - 1):
            assert len(sorted_data[i][0]) >= len(
                sorted_data[i + 1][0]
            ), f"dataset[{i}][0] = {sorted_data[i][0]} is shorter than dataset[{i+1}][0] = {sorted_data[i+1][0]}"

        assert len(sorted_data) == len(
            original_data
        ), f"reordered dataset has length {len(sorted_data)}, should be {len(dataset)}"

        for sorted_data, orignal in zip(original_data, data):
            assert sorted_data == orignal

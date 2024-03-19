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

from typing import Iterator

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler, T_co

from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    GreedyUntilWithLogitsRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
)


class DynamicBatchDataset(Dataset):
    def __init__(
        self,
        requests: list,
        dataset_splits: int,
    ):
        """
        This dataset class uses dynamic batching to speed up the generation.
        Each request is sorted by the length of the prompt + the length of the
        continuation. Then, the dataset is split into dataset_splits splits.
        The first split will contain the longest requests, the second split will
        contain the second longest requests, etc. This allows us to use dynamic
        batching by starting with a small batch size and doubling it for each
        split. This is much faster than using a fixed batch size for the whole
        dataset.

        Args:
            requests (List): A list of requests.
            dataset_splits (int): The number of dataset splits.
        """
        # We make sure the requests contain the tokenized versions of their values
        if any(r.tokenized_context is None for r in requests):
            raise ValueError("You passed a request for which tokenization had not happened yet.")

        # sort the requests using the collate function and save the original order
        enumerated_requests = list(enumerate(requests))
        sorted_enumerated_requests = sorted(enumerated_requests, key=lambda x: self._sorting_criteria(x[1]))

        self.sorted_data = [x[1] for x in sorted_enumerated_requests]
        self.original_order = [x[0] for x in sorted_enumerated_requests]

        self.total_size = len(self.sorted_data)

        if dataset_splits >= self.total_size:
            hlog_warn(
                f"dataset_splits ({dataset_splits}) >= total_size ({self.total_size}), setting dataset_splits to 1"
            )
            dataset_splits = 1

        self.dataset_splits = dataset_splits
        self.split_size = self.total_size // self.dataset_splits + 1
        self.split_start = 0
        self.split_end = min(self.split_start + self.split_size, self.total_size)

    def get_original_order(self, new_arr: list) -> list:
        """
        Get the original order of the data.

        Args:
            newarr (list): Array containing any kind of data that needs to be
                reset in the original order.

        Returns:
            list: new_arr in the original order.
        """
        original_order = [None] * self.total_size

        for original_index, v in zip(self.original_order, new_arr):
            original_order[original_index] = v

        if None in original_order:
            raise RuntimeError(
                f"Some elements of the original order are None, meaning that len(new_arr) ({len(new_arr)}) != len(original_array) ({self.total_size})"
            )

        return original_order

    def get_split_start_end(self, split_id: int) -> tuple[int, int]:
        """
        Get the start and end indices of a dataset split.

        Args:
            split_id (int): The ID of the split.

        Returns:
            tuple: A tuple containing the start and end indices of the split.
        """
        self.split_start = split_id * self.split_size
        self.split_end = min(self.split_start + self.split_size, self.total_size)
        return self.split_start, self.split_end

    def splits_start_end_iterator(self) -> tuple[int, int]:
        """
        Iterator that yields the start and end indices of each dataset split.
        Also updates the starting batch size for each split (trying to double
        the batch every time we move to a new split).

        Yields:
            tuple: A tuple containing the start and end indices of a split.
        """
        for split_id in range(self.dataset_splits):
            yield self.get_split_start_end(split_id)

    def __getitem__(self, index) -> Request:
        """
        Get an item from the dataset depending on the split we are currently in.
        For instance, if we are in split 0, we will get the item at index 0, if
        we are in split 1, we will get the item at index self.split_size, etc.
        Used for dynamic batching.

        Args:
            index (int): The index of the item.

        Returns:
            Any: The item at the specified index.
        """
        return self.sorted_data[index + self.split_start]

    def __len__(self) -> int:
        """
        Get the length of current split the dataset.
        All splits have the same length, except the last one which might be
        shorter.

        Returns:
            int: The length of the dataset.
        """
        return self.split_end - self.split_start

    def _sorting_criteria(self, request) -> int:
        raise NotImplementedError()


class LoglikelihoodDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request: LoglikelihoodRequest | LoglikelihoodRollingRequest) -> int:
        """
        Collates the input data for batching.

        the negative sign on len(toks) sorts descending - this has a few
        advantages:
        - time estimates will always be over not underestimates, which is
        more useful for planning
        - to know the size of a batch when going through the list, you
        know the first one is always the batch padded context length. this
        is useful to simplify the batching logic and more importantly to make
        automatic adaptive batches much much easier to implement
        - any OOMs will happen right away rather than near the end

        Args:
            x (tuple): A tuple containing the input data.

        Returns:
            tuple: A tuple containing the sorted input data.
        """
        toks = request.tokenized_context + request.tokenized_continuation
        return -len(toks)


class LoglikelihoodSingleTokenDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request: LoglikelihoodSingleTokenRequest) -> int:
        """
        Collates the input data for batching.

        the negative sign on len(toks) sorts descending - this has a few # advantages:
        - time estimates will always be over not underestimates, which is
        more useful for planning
        - to know the size of a batch when going through the list, you
        know the first one is always the batch padded context length. this
        is useful to simplify the batching logic and more importantly to make
        automatic adaptive batches much much easier to implement
        - any OOMs will happen right away rather than near the end
        """
        # We take only the prompt, no need for the continuation (since it's a list of single tokens)
        toks = request.tokenized_context
        return -len(toks)


class GenerativeTaskDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request: GreedyUntilRequest | GreedyUntilWithLogitsRequest) -> int:
        """
        Collate function for generating batches.

        Args:
            x (Any): The input data.

        Returns:
            Any: The collated data.
        """
        toks = request.tokenized_context
        gen_length = request.generation_size
        # The generative task has no limit except the model context
        if gen_length is None:
            gen_length = 0
        return -(len(toks) + gen_length)


class GenerativeTaskDatasetNanotron(DynamicBatchDataset):
    def __getitem__(self, index) -> Request:
        """
        Get an item from the dataset depending on the split we are currently in.
        For instance, if we are in split 0, we will get the item at index 0, if
        we are in split 1, we will get the item at index self.split_size, etc.
        Used for dynamic batching.

        Args:
            index (int): The index of the item.

        Returns:
            Any: The item at the specified index.
        """
        return index, self.sorted_data[index + self.split_start]

    def _sorting_criteria(self, request) -> int:
        """
        Collate function for generating batches.

        Args:
            x (Any): The input data.

        Returns:
            Any: The collated data.
        """
        toks = request.tokenized_context
        gen_length = request.generation_size
        return -(len(toks) + gen_length)


class GenDistributedSampler(DistributedSampler):
    """A distributed sampler that copy the last element only when drop_last is False so we keep a small padding in the batches
    as our samples are sorted by length.
    """

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            indices += [
                indices[-1]
            ] * padding_size  # This is our only change here compared to the original DistributedSampler
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

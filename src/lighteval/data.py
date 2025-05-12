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
import math
from typing import Iterator

import torch
from packaging import version
from torch.utils.data import Dataset, Subset


if version.parse(torch.__version__) >= version.parse("2.5.0"):
    from torch.utils.data.distributed import DistributedSampler, _T_co
else:
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data.distributed import T_co as _T_co

from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
)


logger = logging.getLogger(__name__)


class DynamicBatchDataset(Dataset):
    def __init__(
        self,
        requests: list,
        num_dataset_splits: int,
    ):
        """
        This dataset class uses dynamic batching to speed up the generation.
        Each request is sorted by the length of the prompt + the length of the
        continuation. Then, the dataset is split into num_dataset_splits splits.
        The first split will contain the longest requests, the second split will
        contain the second longest requests, etc. This allows us to use dynamic
        batching by starting with a small batch size and doubling it for each
        split. This is much faster than using a fixed batch size for the whole
        dataset.

        Args:
            requests (List): A list of requests.
            num_dataset_splits (int): The number of dataset splits.
        """
        # sort the requests using the collate function and save the original order
        enumerated_requests = list(enumerate(requests))
        sorted_enumerated_requests = sorted(enumerated_requests, key=lambda x: self._sorting_criteria(x[1]))

        self.sorted_data = [x[1] for x in sorted_enumerated_requests]
        self.original_order = [x[0] for x in sorted_enumerated_requests]

        self.total_size = len(self.sorted_data)

        self.num_dataset_splits, self.splits = self.init_split_limits(num_dataset_splits)

    def init_split_limits(self, num_dataset_splits):
        if num_dataset_splits >= self.total_size:
            logger.warning(
                f"num_dataset_splits ({num_dataset_splits}) >= total_size ({self.total_size}), setting num_dataset_splits to 1"
            )
            num_dataset_splits = 1

        split_size = math.ceil(self.total_size / num_dataset_splits)
        splits_indices = [
            (ix * split_size, min((ix + 1) * split_size, self.total_size)) for ix in range(num_dataset_splits)
        ]

        return num_dataset_splits, splits_indices

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

    def splits_iterator(self) -> Iterator[Subset]:
        """
        Iterator that yields the dataset splits based on the split limits.

        Yields:
            Subset: A subset of the dataset.
        """
        split_range = self.num_dataset_splits
        if self.total_size == 0:
            split_range = 0
        for i in range(split_range):
            split_start, split_end = self.splits[i]
            yield Subset(self, range(split_start, split_end))

    def __getitem__(self, index) -> Request:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            Any: The item at the specified index.
        """
        return self.sorted_data[index]

    def __len__(self) -> int:
        """
        Get the length of current split the dataset.
        All splits have the same length, except the last one which might be
        shorter.

        Returns:
            int: The length of the dataset.
        """
        return len(self.sorted_data)

    def __iter__(self) -> Iterator[Request]:
        """
        Iterator that yields the items of the dataset depending on the split we
        are currently in. For instance, if we are in split 0, we will get the
        items from index 0 to self.split_size, if we are in split 1, we will get
        the items from index self.split_size to 2 * self.split_size, etc. Used
        for dynamic batching.

        Yields:
            Any: The items of the dataset.
        """
        for i in range(len(self)):
            yield self.sorted_data[i]

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
    def init_split_limits(self, num_dataset_splits):
        """Initialises the split limits based on generation parameters.
        The splits are used to estimate time remaining when evaluating, and in the case of generative evaluations, to group similar samples together.

        For generative tasks, self._sorting_criteria outputs:
        - a boolean (whether the generation task uses logits)
        - a list (the stop sequences)
        - the item length (the actual size sorting factor).

        In the current function, we create evaluation groups by generation parameters (logits and eos), so that samples with similar properties get batched together afterwards.
        The samples will then be further organised by length in each split.

        Args:
            num_dataset_splits (_type_): _description_

        Returns:
            _type_: _description_
        """
        if num_dataset_splits is not None:
            logger.warning(
                "You cannot select the number of dataset splits for a generative evaluation at the moment. Automatically inferring."
            )

        if len(self.sorted_data) > 0:
            all_sorting_criterion = [self._sorting_criteria(self.sorted_data[0])[:-1]]
        splits_indices = [[0, None]]
        for ix, req in enumerate(self.sorted_data):
            current_sorting_criteria = self._sorting_criteria(req)
            current_key = current_sorting_criteria[:-1]
            if current_key not in all_sorting_criterion:
                all_sorting_criterion.append(current_key)
                splits_indices[-1][1] = ix
                splits_indices.append([ix, None])

        # We add the last split
        splits_indices[-1][1] = self.total_size

        num_dataset_splits = len(splits_indices)
        splits_indices = [tuple(e) for e in splits_indices]
        return num_dataset_splits, splits_indices

    def _sorting_criteria(self, request: GreedyUntilRequest) -> tuple[bool, bool, list, int, int]:
        """
        Collate function for generating batches.

        Args:
            x (Any): The input data.

        Returns:
            Any: The collated data.
        """
        toks = request.context
        gen_length = request.generation_size

        # The generative task has no limit except the model context
        if gen_length is None:
            gen_length = 0

        return (
            request.do_sample,
            request.use_logits,
            tuple(request.stop_sequence),
            gen_length,
            -(len(toks) + gen_length),
        )


class GenerativeTaskDatasetNanotron(GenerativeTaskDataset):
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


class GenDistributedSampler(DistributedSampler):
    """A distributed sampler that copy the last element only when drop_last is False so we keep a small padding in the batches
    as our samples are sorted by length.
    """

    def __iter__(self) -> Iterator[_T_co]:
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

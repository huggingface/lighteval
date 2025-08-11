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

import functools
import hashlib
import json
import logging
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Callable, List, Set, Tuple, Union

import pandas as pd
from datasets import Dataset, load_dataset

from lighteval.models.abstract_model import ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


class SampleType(Enum):
    PREDICTIONS = 1
    TOKENIZED_INPUTS = 2  # Not implemented yet


class SampleCache:
    """
    Disk-based cache for sample evaluation results using HuggingFace datasets.
    The model hash is a hash of the model config, to make sure we rerun the eval if any parameter changes
    (generation param, model version, etc).

    Cache Structure:
    - {cache_dir}/
      - {sample_type}/
        - {model_name}/
          - {model_hash}/
            - {task_name}.parquet
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the sample cache.

        Args:
            model_config: Configuration for the model being cached
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(os.path.expanduser(model_config.cache_dir))
        self.model_config = model_config
        self.model_hash = self.get_model_hash(model_config)

        # Create cache directory structure and load cached indices if present
        self.all_cache_dirs = {}
        self.existing_indices = {}
        for sample_type in SampleType:
            self.all_cache_dirs[sample_type] = (
                self.cache_dir / sample_type.name.lower() / self.model_config.model_name / self.model_hash
            )
            self.all_cache_dirs[sample_type].mkdir(parents=True, exist_ok=True)
            self.existing_indices[sample_type] = self._get_cached_indices(sample_type)

    def _get_cached_indices(self, sample_type: SampleType) -> dict:
        """Loads all indices for samples which are properly cached

        Returns:
            dict: Dictionary mapping task names to lists of cached sample indices
        """
        cached_indices = {}
        cache_dir = self.all_cache_dirs[sample_type]

        if not cache_dir.exists():
            return cached_indices

        for cache_file in cache_dir.glob("*.parquet"):
            task_name = cache_file.stem
            try:
                dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                sample_ids = []
                for row in dataset:
                    try:
                        # We only save indices of correctly formatted samples, though this means we need to load each at least once
                        self._load_sample(row, sample_type=sample_type)
                        sample_ids.append(row["sample_id"])
                    except Exception:
                        continue

                cached_indices[task_name] = sample_ids
                logger.debug(f"Loaded {len(sample_ids)} cached indices for task '{task_name}' from {cache_file}")
            except Exception as e:
                logger.warning(f"Error loading cached indices for task '{task_name}' from {cache_file}: {e}")

        return cached_indices

    def get_model_hash(self, model_config: ModelConfig) -> str:
        """Create a hash for model configuration."""
        # Use Pydantic's model_dump instead of asdict for BaseModel
        config_dict = model_config.model_dump()
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_cache_path(self, task_name: str, sample_type: SampleType) -> Path:
        """Get the file path for a specific task's cache file.

        Args:
            task_name: Name of the task
            sample_type: Type of samples being cached

        Returns:
            Path: Path to the cache file for the given task and sample type
        """
        return self.all_cache_dirs[sample_type] / f"{task_name}.parquet"

    def _load_sample(
        self, sample: pd.core.series.Series | dict, sample_type: SampleType
    ) -> Union[dict, ModelResponse]:
        """Load a sample from cached data based on sample type.

        Args:
            sample: Raw sample data from cache, arrives as a dataframe row
            sample_type: Type of sample being loaded

        Returns:
            Union[dict, ModelResponse]: Loaded sample in appropriate format for processing
        """
        # If we just use the pandas dict, lists are converted to np arrays which we don't want
        if isinstance(sample, pd.core.series.Series):
            sample = json.loads(sample.to_json())
        if sample_type == SampleType.TOKENIZED_INPUTS:
            return sample["sample"]
        elif sample_type == SampleType.PREDICTIONS:
            return ModelResponse(**sample["sample"])

    def _dump_sample(self, result: Union[dict, ModelResponse], sample_type: SampleType) -> dict:
        """Dumps the sample in the correct format for file saving

        Args:
            result (Union[dict, ModelResponse]): Processed sample to save
            sample_type (SampleType): Type of sample

        Returns:
            dict
        """
        if sample_type == SampleType.TOKENIZED_INPUTS:
            return result
        elif sample_type == SampleType.PREDICTIONS:
            return asdict(result)

    def get_notcached_samples(self, docs: List[Doc], sample_type: SampleType) -> Tuple[List[Doc], Set]:
        """
        Identify which docs need processing based on cached indices.

        Returns:
            Tuple of (docs_not_cached, tasks_with_cached_samples) where
                - docs_not_cached contains docs that need processing
                - tasks_with_cached_samples are the tasks that have some cached samples
        """
        cached_indices = self.existing_indices[sample_type]

        docs_not_cached = []
        tasks_with_cached_samples = set()

        for doc in docs:
            task_name = doc.task_name
            if task_name in cached_indices and doc.id in cached_indices[task_name]:
                tasks_with_cached_samples.add(task_name)
            else:
                docs_not_cached.append(doc)

        return docs_not_cached, set(tasks_with_cached_samples)

    def get_samples_from_cache(
        self, docs: List[Doc], task_names: list | set, sample_type: SampleType
    ) -> List[dict | ModelResponse]:
        """
        Get cached samples for the given docs.
        Warning: Assumes all docs and task_names provided are stored in cache, will fail otherwise.

        Returns:
            List of cached items
        """
        # Load datasets for tasks that have cached docs
        task_datasets = {}

        for task_name in task_names:
            cache_file = self.get_cache_path(task_name=task_name, sample_type=sample_type)
            try:
                dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                dataset_df = dataset.to_pandas().set_index("sample_id")
                task_datasets[task_name] = dataset_df
            except Exception as e:
                logger.warning(f"Error loading {sample_type.name.lower()} cache for {task_name}: {e}")

        # Build results list
        results = []

        for doc in docs:
            row = task_datasets[doc.task_name].loc[doc.id]
            results.append(self._load_sample(row, sample_type))

        return results

    def store_samples(
        self,
        docs: List[Doc],
        results: List[dict] | List[ModelResponse],
        task_names: list[str],
        sample_type: SampleType,
    ):
        """Store new results for samples in docs"""
        if not results:
            return

        # Prepare newly processed data for dataset
        processed_data = {task_name: [] for task_name in task_names}
        for doc, result in zip(docs, results):
            processed_data[doc.task_name].append(
                {"sample_id": doc.id, "sample": self._dump_sample(result, sample_type)}
            )
        processed_data = {task_name: task_data for task_name, task_data in processed_data.items() if task_data}

        # Concatenate it with existing data and save to file
        for task_name, task_data in processed_data.items():
            cache_file = self.get_cache_path(task_name=task_name, sample_type=sample_type)

            # Load existing data if present
            existing_data = []
            if cache_file.exists():
                try:
                    existing_dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                    existing_data = existing_dataset.to_list()
                except Exception as e:
                    logger.error(f"Error loading existing {sample_type.name.lower()} cache for {task_name}: {e}")

            # Merge with new data (new data overwrites existing)
            existing_ids = {row["sample_id"] for row in existing_data}

            if any(row["sample_id"] in existing_ids for row in task_data):
                logger.warning(
                    "Unexpected behavior: You have reprocessed already cached items - we will ignore the new version."
                )
            all_samples = existing_data + [row for row in task_data if row["sample_id"] not in existing_ids]

            # Save updated dataset
            dataset = Dataset.from_list(all_samples)
            dataset.to_parquet(str(cache_file))

            logger.info(
                f"Cached {len(all_samples)} {sample_type.name.lower()} samples of {task_name} at {str(cache_file)}."
            )

            # Refresh cached indices after storing new samples
            self.existing_indices[sample_type][task_name] = [sample["sample_id"] for sample in all_samples]


def cached(cache_type_name: str):  # noqa C901
    """
    Decorator to cache method results based on Doc inputs.

    Args:
        cache_type_name: Type of cache ("tokenization" or "predictions")

    Usage:
        @cached("tokenization")
        def tok_encode_pair(self, docs: List[Doc], ...):
            # method implementation

        @cached("predictions")
        def greedy_until(self, docs: List[Doc], ...):
            # method implementation
    """

    def decorator(func: Callable):  # noqa C901
        @functools.wraps(func)
        def wrapper(self, docs: Union[Doc, List[Doc]], *args, **kwargs):  # noqa C901
            cache_type = SampleType[cache_type_name.upper()]
            docs = as_list(docs)

            # Check if caching is enabled for the model
            if not hasattr(self, "_cache") or self._cache is None:
                return func(self, docs, *args, **kwargs)

            cache: SampleCache = self._cache

            # Extract task names
            task_names = {doc.task_name for doc in docs}

            # 1) Identify which samples must be processed because they are not cached
            docs_not_cached, tasks_with_cached_samples = cache.get_notcached_samples(docs, cache_type)

            # Log cache statistics
            cached_count = len(docs) - len(docs_not_cached)
            if cached_count > 0:
                logger.info(
                    f"Cache: {cached_count}/{len(docs)} {cache_type.name.lower()} samples are cached for tasks {', '.join(tasks_with_cached_samples)}"
                )

            # 2) Process not cached docs and save to file
            new_results = []
            if docs_not_cached:
                notcached_task_names = {doc.task_name for doc in docs_not_cached}
                logger.info(
                    f"Cache: Processing {len(docs_not_cached)}/{len(docs)} {cache_type.name.lower()} samples for tasks {', '.join(notcached_task_names)}"
                )
                new_results = func(self, docs_not_cached, *args, **kwargs)

                # Store new results in file cache
                cache.store_samples(
                    docs=docs_not_cached, results=new_results, task_names=task_names, sample_type=cache_type
                )

            # 3) Create final results by pulling from newly saved file cache
            final_results = cache.get_samples_from_cache(docs, task_names, cache_type)

            if any(r is None for r in final_results):
                raise ValueError("Problem while loading and aggregating items from cache.")

            return final_results

        return wrapper

    return decorator

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
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Set, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict

from lighteval.models.abstract_model import ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


class SampleCache:
    """Disk-based cache for sample evaluation results using HuggingFace datasets.
    The model hash is a hash of the model config, to make sure we rerun the eval if any parameter changes
    (generation param, model version, etc).

    Cache Structure:
    - {cache_dir}/
        - {model_name}/
            - {model_hash}/
                - {task_name}/
                    - {task_hash}/ dataset dict, where splits are SamplingMethod
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize the sample cache.

        Args:
            model_config: Configuration for the model being cached
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(os.path.expanduser(model_config.cache_dir))
        self.model_config = model_config
        self.model_hash = self.get_model_hash(model_config)

        self.registry = None

        # Create cache directory structure and load cached indices if present
        self.all_cache_dirs = {}
        self.existing_indices = {}
        self.all_cache_dirs = self.cache_dir / self.model_config.model_name / self.model_hash
        self.all_cache_dirs.mkdir(parents=True, exist_ok=True)
        # (task_name, task_hash, sampling_method)
        self.existing_indices = self._get_cached_indices()

    def _init_registry(self, registry: Registry):
        self.registry = registry

    def _get_cached_indices(self) -> dict:
        """Loads all indices for samples which are properly cached

        Returns:
            dict: Dictionary mapping task names to lists of cached sample indices
        """
        cached_indices = {}
        cache_dir = self.all_cache_dirs

        if not cache_dir.exists():
            return cached_indices

        for cache_file in cache_dir.rglob("*.parquet"):
            task_name = str(cache_file.parent).split("/")[-1]
            task_hash = cache_file.stem
            try:
                full_dataset = DatasetDict.load_from_disk(str(cache_file))
                for sampling_method in [SamplingMethod.GENERATIVE, SamplingMethod.LOGPROBS]:
                    sample_ids = []
                    for row in full_dataset[str(sampling_method)]:
                        try:
                            # We only save indices of correctly formatted samples, though this means we need to load each at least once
                            self._load_sample(row)
                            cur_sample = row["sample_id"]
                            sample_ids.append(cur_sample)
                        except Exception:
                            continue

                    cached_indices[(task_name, task_hash, sampling_method)] = sample_ids
                    logger.debug(
                        f"Loaded {len(sample_ids)} cached indices for task '{task_name}', {str(sampling_method)} from {cache_file}"
                    )
            except Exception as e:
                logger.warning(f"Error loading cached indices for task '{task_name}' from {cache_file}: {e}")

        return cached_indices

    def get_model_hash(self, model_config: ModelConfig) -> str:
        """Create a hash for model configuration.

        Returns:
            str: A 16-character hexadecimal hash of the model configuration
        """
        # Use Pydantic's model_dump instead of asdict for BaseModel
        config_dict = model_config.model_dump()
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_task_hash(self, full_task_name: str) -> str:
        if self.registry is None:
            logger.warning(
                "The task registry was not provided to the cache config. We can't test if the current task has the same hash as the saved tasks."
            )
            return "NO_HASH"
        task_suite, task_name, _ = full_task_name.split("|")
        task_configs: list[LightevalTaskConfig] = sorted(self.registry.task_to_configs[f"{task_suite}|{task_name}"])
        config_str = "|".join([task_config.__str__(lite=True) for task_config in task_configs])
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_cache_path(self, task_name: str, task_hash: str) -> Path:
        """Get the file path for a specific task's cache file.

        Args:
            task_name: Name of the task
            task_hash: Hash of the task config, obtainable with self.get_task_hash
            sample_type: Type of samples being cached

        Returns:
            Path: Path to the cache file for the given task and sample type
        """
        return self.all_cache_dirs / task_name / task_hash

    def get_sampling_method(self, sample: dict) -> str:
        if len(sample.get("logprobs", [])) > 0:
            return SamplingMethod.LOGPROBS
        if len(sample.get("text", [])) > 0:
            return SamplingMethod.GENERATIVE
        return None

    def _load_sample(self, sample: pd.core.series.Series | dict) -> Union[dict, ModelResponse]:
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
        return ModelResponse(**sample["sample"])

    def _dump_sample(self, result: Union[dict, ModelResponse]) -> dict:
        """Dumps the sample in the correct format for file saving

        Args:
            result (Union[dict, ModelResponse]): Processed sample to save

        Returns:
            dict
        """
        return asdict(result)

    def get_notcached_samples(self, docs: List[Doc], sampling_method: SamplingMethod) -> Tuple[List[Doc], Set]:
        """
        Identify which docs need processing based on cached indices.

        Returns:
            Tuple of (docs_not_cached, tasks_with_cached_samples) where
                - docs_not_cached contains docs that need processing
                - tasks_with_cached_samples are the tasks that have some cached samples
        """
        cached_indices = self.existing_indices

        docs_not_cached = []
        tasks_with_cached_samples = set()

        for doc in docs:
            task_name = doc.task_name
            task_hash = self.get_task_hash(task_name)
            task_id = (task_name, task_hash, sampling_method)
            try:
                if doc.id in cached_indices[task_id][sampling_method]:
                    tasks_with_cached_samples.add(task_id)
                else:
                    docs_not_cached.append(doc)
            except KeyError:  # task id or sampling method not yet there
                docs_not_cached.append(doc)

        return docs_not_cached, set(tasks_with_cached_samples)

    def get_samples_from_cache(
        self, docs: List[Doc], task_ids: list | set, sampling_method: SamplingMethod
    ) -> List[dict | ModelResponse]:
        """Get cached samples for the given docs.
        Warning: Assumes all docs and task_names provided are stored in cache, will fail otherwise.

        Returns:
            List of cached items
        """
        # Load datasets for tasks that have cached docs
        task_datasets = {}

        for task_name, task_hash, task_sampling_method in task_ids:
            if task_sampling_method != sampling_method:
                continue
            cache_file = self.get_cache_path(task_name=task_name, task_hash=task_hash)
            try:
                dataset = DatasetDict.load_from_disk(str(cache_file))[str(sampling_method)]
                dataset_df = dataset.to_pandas().set_index("sample_id")
                task_datasets[(task_name, task_hash, sampling_method)] = dataset_df
            except Exception as e:
                logger.warning(f"Error loading prediction cache for {task_name}: {e}")

        # Build results list
        results = []

        for doc in docs:
            task_name = doc.task_name
            task_hash = self.get_task_hash(task_name)
            task_id = (task_name, task_hash, sampling_method)
            row = task_datasets[task_id].loc[doc.id]
            results.append(self._load_sample(row))

        return results

    def store_samples(  # noqa C901
        self,
        docs: List[Doc],
        results: List[dict] | List[ModelResponse],
        task_ids: list[tuple[str, str]],
        sampling_method: SamplingMethod,
    ):
        """Store new results for samples in docs"""
        if not results:
            return

        # Prepare newly processed data for dataset
        processed_data = {task_id: [] for task_id in task_ids}
        for doc, result in zip(docs, results):
            task_name = doc.task_name
            task_hash = self.get_task_hash(task_name)
            task_id = (task_name, task_hash, sampling_method)
            sample = self._dump_sample(result)

            if self.get_sampling_method(sample) != sampling_method:
                logger.warning("The sample which was returned by the model is not of the expected type ")

            processed_data[task_id].append({"sample_id": doc.id, "sample": sample})
        processed_data = {task_id: task_data for task_id, task_data in processed_data.items() if task_data}

        # Concatenate it with existing data and save to file
        for (task_name, task_hash, sampling_method), task_data in processed_data.items():
            if (task_name, task_hash, sampling_method) not in self.existing_indices.keys():
                self.existing_indices[(task_name, task_hash, sampling_method)] = {}

            cache_file = self.get_cache_path(task_name=task_name, task_hash=task_hash)

            # Load existing data if present
            existing_data = []
            existing_samples = {}
            if cache_file.exists():
                try:
                    existing_dataset = DatasetDict.load_from_disk(str(cache_file))[str(sampling_method)]
                    existing_data = existing_dataset.to_list()
                except KeyError:
                    logger.info(f"No data was cached for {task_name} ({task_hash}, {str(sampling_method)}")
                except Exception as e:
                    logger.error(
                        f"Error loading existing prediction cache for {task_name} ({task_hash}, {str(sampling_method)}): {e}"
                    )

                existing_samples = {
                    (row["sample_id"], self.get_sampling_method(row["sample"])) for row in existing_data
                }
                if any(
                    (row["sample_id"], self.get_sampling_method(row["sample"])) in existing_samples
                    for row in task_data
                ):
                    logger.warning(
                        "Unexpected behavior: You have reprocessed already cached items - we will ignore the new version."
                    )

            # Merge with new data (new data overwrites existing)
            # We look at id + sampling method
            new_data = [
                row
                for row in task_data
                if (row["sample_id"], self.get_sampling_method(row["sample"])) not in existing_samples
            ]
            all_samples = existing_data + new_data

            # Check if file exists and has other configs we need to preserve
            dataset_dict = {}
            if cache_file.exists():
                try:
                    # We load in memory to overwrite the written file
                    dataset_dict = DatasetDict.load_from_disk(str(cache_file), keep_in_memory=True)
                except Exception as e:
                    logger.debug(f"Could not load existing configs from {cache_file}: {e}")

            # Add our current config, we overwrite the existing
            dataset = Dataset.from_list(all_samples)
            dataset_dict[str(sampling_method)] = dataset

            # Save as DatasetDict to preserve all configs
            full_dataset = DatasetDict(dataset_dict)
            if cache_file.exists():
                shutil.rmtree(cache_file)
            full_dataset.save_to_disk(str(cache_file))

            logger.info(f"Cached {len(all_samples)} samples of {task_name} at {str(cache_file)}.")

            # Refresh cached indices after storing new samples
            self.existing_indices[(task_name, task_hash, sampling_method)] = [
                sample["sample_id"] for sample in all_samples
            ]


def cached(sampling_method: SamplingMethod = None):  # noqa C901
    """
    Decorator to cache method results based on Doc inputs.

    Args:
        cache_type_name: Type of cache ("tokenization" or "predictions")

    Usage:
        @cached("greedy")
        def greedy_until(self, docs: List[Doc], ...):
            # method implementation

    Returns:
        Callable: A decorator function that wraps the original function with caching functionality
    """

    def decorator(func: Callable):  # noqa C901
        @functools.wraps(func)
        def wrapper(self, docs: Union[Doc, List[Doc]], *args, **kwargs):  # noqa C901
            docs = as_list(docs)

            # Check if caching is enabled for the model
            if not hasattr(self, "_cache") or self._cache is None:
                return func(self, docs, *args, **kwargs)

            cache: SampleCache = self._cache

            # Extract task names
            task_ids = {(doc.task_name, cache.get_task_hash(doc.task_name), sampling_method) for doc in docs}

            # 1) Identify which samples must be processed because they are not cached
            docs_not_cached, tasks_with_cached_samples = cache.get_notcached_samples(docs, sampling_method)

            # Log cache statistics
            cached_count = len(docs) - len(docs_not_cached)
            if cached_count > 0:
                logger.info(
                    f"Cache: {cached_count}/{len(docs)} samples are cached for tasks {', '.join(t[0] for t in tasks_with_cached_samples)}"
                )

            # 2) Process not cached docs and save to file
            new_results = []
            if docs_not_cached:
                notcached_task_names = {(doc.task_name, cache.get_task_hash(doc.task_name)) for doc in docs_not_cached}
                notcached_task_names_str = ", ".join(
                    f"{task_name} ({task_hash})" for task_name, task_hash in notcached_task_names
                )
                logger.info(
                    f"Cache: Processing {len(docs_not_cached)}/{len(docs)} samples for tasks {notcached_task_names_str}"
                )
                new_results = func(self, docs_not_cached, *args, **kwargs)

                # Store new results in file cache
                cache.store_samples(
                    docs=docs_not_cached,
                    results=new_results,
                    task_ids=task_ids,
                    sampling_method=sampling_method,
                )

            # 3) Create final results by pulling from newly saved file cache
            final_cached_results = cache.get_samples_from_cache(docs, task_ids, sampling_method)

            # 4) We only keep samples with the correct sampling method
            final_results = [
                s for s in final_cached_results if cache.get_sampling_method(cache._dump_sample(s)) == sampling_method
            ]

            if any(r is None for r in final_results):
                raise ValueError("Problem while loading and aggregating items from cache.")

            return final_results

        return wrapper

    return decorator

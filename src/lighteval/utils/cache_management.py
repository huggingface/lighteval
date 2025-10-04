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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Set, Tuple, Union

import pandas as pd
from datasets import Dataset, load_dataset

from lighteval.models.abstract_model import ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


@dataclass
class TaskID:
    """A unique ID for a grouping of task samples. It relies on the task name,
    the task config (which gives the task_hash), and the sampling method (linked to
    the metric type)
    """

    task_name: str
    task_hash: str
    sampling_method: SamplingMethod

    def __str__(self):
        return f"{self.task_name} ({self.task_hash}, {self.sampling_method.name})"

    def __hash__(self):
        return int.from_bytes(hashlib.sha256(str(self).encode()).digest(), byteorder="big")


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
        self.model_config = model_config
        self.model_hash = self.get_model_hash(model_config)

        self.cache_dir = (
            Path(os.path.expanduser(self.model_config.cache_dir)) / self.model_config.model_name / self.model_hash
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.registry = None

        self.existing_indices = self._load_cached_indices()
        # Caching the task_hashes to avoid grabbing the registry all the time
        self._task_hashes = {}

    def _init_registry(self, registry: Registry):
        self.registry = registry

    def _load_cached_indices(self) -> dict:
        """Loads all indices for samples which are properly cached. We recursively search for all available tasks and files.

        Returns:
            dict: Dictionary mapping task names to lists of cached sample indices
        """
        logger.info("[CACHING] Initializing data cache")
        cached_indices = {}
        cache_dir = self.cache_dir

        if not cache_dir.exists():
            return cached_indices

        for cache_file in cache_dir.rglob("*.parquet"):
            try:
                # cache_file.parts gives all the subfolders of the url, up to the file name
                # last 3 are task_name/task_hash/file_name.parquet, so we take -3 and -2
                task_name, task_hash = cache_file.parts[-3:-1]
                sampling_method = SamplingMethod[cache_file.stem]  # removes the file extension
                task_id = TaskID(task_name, task_hash, sampling_method)

                full_dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                sample_ids = []
                for row in full_dataset:
                    try:
                        # We only save indices of correctly formatted samples, though this means we need to load each at least once
                        self._load_sample(row)
                        cur_sample = row["sample_id"]
                        sample_ids.append(cur_sample)
                    except Exception:
                        continue

                cached_indices[task_id] = sample_ids
                logger.info(
                    f"[CACHING] Loaded {len(sample_ids)} cached indices for task '{str(task_id)} from {cache_file}"
                )
            except Exception as e:
                logger.warning(f"Error loading cached indices from {cache_file}: {e}")

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

    def _get_task_hash(self, full_task_name: str) -> str:
        """Builds a task_hash from the LightevalTaskConfig loaded from the task name and the registry.

        Args:
            full_task_name (str): task_name as provided to the registry (with suite|task|few_shot)

        Returns:
            str: a hash of the task config in its current state in the registry, or the NO_HASH string if the
            registry has not been preloaded
        """
        if self.registry is None:
            logger.warning(
                "The task registry was not provided to the cache config. We can't test if the current task has the same hash as the saved tasks."
            )
            return "NO_HASH"
        if full_task_name not in self._task_hashes:
            task_suite, task_name, _ = full_task_name.split("|")
            task_configs: list[LightevalTaskConfig] = sorted(
                self.registry.task_to_configs[f"{task_suite}|{task_name}"]
            )
            config_str = "|".join([task_config.__str__(lite=True) for task_config in task_configs])
            task_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            self._task_hashes[full_task_name] = task_hash
        return self._task_hashes[full_task_name]

    def get_cache_path(self, task_id: TaskID) -> Path:
        """Get the file path for a specific task's cache file.

        Args:
            task_id: TaskID of the task

        Returns:
            Path: Path to the cache file for the given task and sample type
        """
        return self.cache_dir / task_id.task_name / task_id.task_hash / f"{task_id.sampling_method.name}.parquet"

    def get_task_id(self, task_name: str, sampling_method: SamplingMethod) -> TaskID:
        """Returns a unique task indentifier. Depends on the task name,
        task version and parameters (from which a hash is derived), and
        current sampling method (current metric we look at).

        Args:
            task_name (str): Name of the task
            sampling_method (SamplingMethod): Sampling used for the current metric

        Returns:
            TaskID: A unique identifier for the task
        """
        task_hash = self._get_task_hash(task_name)
        return TaskID(task_name, task_hash, sampling_method)

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

    def get_samples_to_process_and_cache(
        self, docs: List[Doc], sampling_method: SamplingMethod
    ) -> Tuple[List[Doc], Set[TaskID]]:
        """
        Identify which docs need processing because they are not cached yet, based on cached doc and task indices.

        Returns:
            Tuple of (docs_not_cached, tasks_with_cached_samples) where
                - docs_not_cached contains docs that need processing
                - tasks_with_cached_samples are the tasks that have some cached samples
        """
        cached_indices = self.existing_indices

        docs_not_cached = []
        tasks_with_cached_samples = set()

        for doc in docs:
            task_id = self.get_task_id(doc.task_name, sampling_method)
            try:
                if doc.id in cached_indices[task_id]:
                    tasks_with_cached_samples.add(task_id)
                else:
                    docs_not_cached.append(doc)
            except KeyError:  # task id or sampling method not yet there
                docs_not_cached.append(doc)

        return docs_not_cached, set(tasks_with_cached_samples)

    def get_samples_from_cache(
        self, docs: List[Doc], task_ids: List[TaskID] | set[TaskID], sampling_method: SamplingMethod
    ) -> List[dict | ModelResponse]:
        """Get cached samples for the given docs.
        Warning: Assumes all docs and task_names provided are stored in cache, will fail otherwise.

        Returns:
            List of cached items
        """
        # Load datasets for tasks that have cached docs
        task_datasets = {}

        for task_id in task_ids:
            if task_id.sampling_method != sampling_method:
                continue
            cache_file = self.get_cache_path(task_id)
            try:
                dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                dataset_df = dataset.to_pandas().set_index("sample_id")
                task_datasets[task_id] = dataset_df
            except Exception as e:
                logger.warning(f"Error loading prediction cache for {str(task_id)}: {e}")

        # Build results list
        results = []

        for doc in docs:
            task_id = self.get_task_id(doc.task_name, sampling_method)
            row = task_datasets[task_id].loc[doc.id]
            results.append(self._load_sample(row))

        return results

    def cache_samples(  # noqa C901
        self,
        docs: List[Doc],
        results: List[dict] | List[ModelResponse],
        task_ids: list[TaskID],
        sampling_method: SamplingMethod,
    ):
        """Store new results for samples in docs"""
        if not results:
            return

        # Prepare newly processed data for dataset
        processed_data = {task_id: [] for task_id in task_ids}
        for doc, result in zip(docs, results):
            task_id = self.get_task_id(doc.task_name, sampling_method)
            sample = self._dump_sample(result)

            processed_data[task_id].append({"sample_id": doc.id, "sample": sample})
        processed_data = {task_id: task_data for task_id, task_data in processed_data.items() if task_data}

        # Concatenate it with existing data and save to file
        for task_id, task_data in processed_data.items():
            if task_id not in self.existing_indices.keys():
                self.existing_indices[task_id] = {}

            cache_file = self.get_cache_path(task_id)

            # Load existing data if present
            existing_data = []
            existing_samples = {}
            if cache_file.exists():
                try:
                    existing_dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                    existing_data = existing_dataset.to_list()
                except KeyError:
                    logger.info(f"No data was cached for {str(task_id)}")
                except Exception as e:
                    logger.error(f"Error loading existing prediction cache for {str(task_id)}: {e}")

                existing_samples = {(row["sample_id"], sampling_method) for row in existing_data}
                if any((row["sample_id"], sampling_method) in existing_samples for row in task_data):
                    logger.warning(
                        "Unexpected behavior: You have reprocessed already cached items - we will ignore the new version."
                    )

            # Merge with new data (new data overwrites existing)
            # We look at id + sampling method
            new_data = [row for row in task_data if (row["sample_id"], sampling_method) not in existing_samples]
            all_samples = existing_data + new_data

            # Save updated dataset
            dataset = Dataset.from_list(all_samples)
            dataset.to_parquet(str(cache_file))

            logger.info(f"Cached {len(all_samples)} samples of {str(task_id)} at {str(cache_file)}.")

            # Refresh cached indices after storing new samples
            self.existing_indices[task_id] = [sample["sample_id"] for sample in all_samples]


def cached(sampling_method: SamplingMethod = None):  # noqa C901
    """
    Decorator to cache method results based on Doc inputs.

    Args:
        cache_type_name: Type of cache ("tokenization" or "predictions")

    Usage:
        @cached(SamplingMethod.GENERATIVE)
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
            task_ids = {cache.get_task_id(doc.task_name, sampling_method) for doc in docs}

            # 1) Identify which samples must be processed because they are not cached
            docs_not_cached: List[Doc]
            tasks_with_cached_samples: Set[TaskID]
            docs_not_cached, tasks_with_cached_samples = cache.get_samples_to_process_and_cache(docs, sampling_method)

            # Log cache statistics
            cached_count = len(docs) - len(docs_not_cached)
            if cached_count > 0:
                logger.info(
                    f"Cache: {cached_count}/{len(docs)} samples are cached for tasks {', '.join(t_id.task_name for t_id in tasks_with_cached_samples)}"
                )

            # 2) Process not cached docs and save to file
            new_results = []
            if docs_not_cached:
                tasks_needing_sample_processing = {
                    cache.get_task_id(doc.task_name, sampling_method) for doc in docs_not_cached
                }
                logger.info(
                    f"Cache: Starting to process {len(docs_not_cached)}/{len(docs)} samples (not found in cache) for tasks {','.join(str(t) for t in tasks_needing_sample_processing)}"
                )
                new_results = func(self, docs_not_cached, *args, **kwargs)

                # Store new results in file cache
                cache.cache_samples(
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

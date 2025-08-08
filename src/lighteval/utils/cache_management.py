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
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union

from datasets import Dataset, load_dataset

from lighteval.models.abstract_model import ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


class SampleType(Enum):
    PREDICTIONS = 1
    TOKENIZED_INPUTS = 2


class SampleCache:
    """
    Disk-based cache for sample evaluation results using HuggingFace datasets.

    This cache stores tokenization and prediction results with simple structure:
    - Each model gets its own subdirectory
    - Each task gets its own parquet file within the model directory
    - Each sample is a single row in the dataset

    Cache Structure:
    - {cache_dir}/
      - tokenization/
        - {model_name}/{model_hash}/
          - {task_name}.parquet
      - predictions/
        - {model_name}/{model_hash}/
          - {task_name}.parquet
    """

    def __init__(self, model_config: ModelConfig, cache_dir: str = "./cache/huggingface/lighteval"):
        """
        Initialize the sample cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.model_config = model_config
        self.model_hash = self.get_model_hash(model_config)

        # Create cache directory structure
        self.tokenization_dir = self.cache_dir / "tokenization" / self.model_config.model_name / self.model_hash
        self.predictions_dir = self.cache_dir / "predictions" / self.model_config.model_name / self.model_hash

        self.tokenization_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def get_model_hash(self, model_config: ModelConfig) -> str:
        """Create a hash for model configuration."""
        # Use Pydantic's model_dump instead of asdict for BaseModel
        config_dict = model_config.model_dump()
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_cache_path(self, task_name: str, sample_type: SampleType) -> str:
        if sample_type == SampleType.TOKENIZED_INPUTS:
            cache_file = self.tokenization_dir / f"{task_name}.parquet"
        elif sample_type == SampleType.PREDICTIONS:
            cache_file = self.predictions_dir / f"{task_name}.parquet"
        return cache_file

    def get_samples(
        self, docs: List[Doc], task_names: list[str], sample_type: SampleType
    ) -> Tuple[List[Optional[dict]], List[Doc], Set]:
        """
        Get cached tokenization results for docs.

        Returns:
            Tuple of (cached_results, uncached_docs, cached_task_names) where
                - cached_results[i] is None if docs[i] is not cached
                - uncached_docs contains docs that need processing
                - cached_task_names are the tasks from which we successfully loaded items
        """
        task_datasets = dict.fromkeys(task_names)
        for task_name in task_names:
            cache_file = self.get_cache_path(task_name=task_name, sample_type=sample_type)

            if cache_file.exists():
                try:
                    dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                    dataset_df = dataset.to_pandas().set_index("sample_id")
                    task_datasets[task_name] = dataset_df
                except Exception as e:
                    logger.warning(f"Error loading {sample_type.name.lower()} cache for {task_name}: {e}")

        results = []
        uncached_docs = []
        cached_task_names = []

        for doc in docs:
            try:
                row = task_datasets[doc.task_name].loc[doc.id]
                if sample_type == SampleType.TOKENIZED_INPUTS:
                    results.append(row["sample"])
                elif sample_type == SampleType.PREDICTIONS:
                    results.append(ModelResponse(**row["sample"]))
                cached_task_names.append(doc.task_name)
            # AttributeError -> we did not manage to load the dataset, so task_datasets is None

            except (AttributeError, KeyError, IndexError):
                results.append(None)
                uncached_docs.append(doc)

        return results, uncached_docs, set(cached_task_names)

    def store_samples(
        self,
        docs: List[Doc],
        results: List[dict] | List[ModelResponse],
        task_names: list[str],
        sample_type: SampleType,
    ):
        """Store tokenization results for docs."""
        if not results:
            return

        # Prepare data for dataset
        tasks_data = {task_name: [] for task_name in task_names}
        for doc, result in zip(docs, results):
            if sample_type == SampleType.TOKENIZED_INPUTS:
                tasks_data[doc.task_name].append({"sample_id": doc.id, "sample": result})
            elif sample_type == SampleType.PREDICTIONS:
                tasks_data[doc.task_name].append({"sample_id": doc.id, "sample": asdict(result)})

        for task_name in task_names:
            # Save dataset
            cache_file = self.get_cache_path(task_name=task_name, sample_type=sample_type)

            # Load existing data if present
            existing_data = []
            if cache_file.exists():
                try:
                    existing_dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                    existing_data = existing_dataset.to_list()
                except Exception as e:
                    logger.warning(f"Error loading existing {sample_type.name.lower()} cache for {task_name}: {e}")

            # Merge with new data (new data overwrites existing)
            existing_ids = {row["sample_id"] for row in existing_data}
            task_data = tasks_data[task_name]

            if any(row["sample_id"] in existing_ids for row in task_data):
                logger.warning(f"Careful, you are overwriting samples for task {task_name}.")
            all_data = existing_data + [row for row in task_data if row["sample_id"] not in existing_ids]

            # Save updated dataset
            dataset = Dataset.from_list(all_data)
            dataset.to_parquet(str(cache_file))

            logger.info(
                f"Cached {len(all_data)} {sample_type.name.lower()} samples of {task_name} at {str(cache_file)}."
            )


def cached(cache_type_name: str):  # noqa C901
    """
    Decorator to cache method results based on Doc inputs.

    Args:
        cache_type: Type of cache ("tokenization" or "predictions")

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

            # Extract task name
            task_names = {doc.task_name for doc in docs}
            results, uncached_docs, cached_tasks_names = cache.get_samples(docs, task_names, cache_type)

            # Process uncached docs if any
            cached_results = [c for c in results if c is not None]
            if cached_results:
                logger.info(
                    f"Cache: Found {len(cached_results)}/{len(docs)} {cache_type.name.lower()} samples for tasks {', '.join(cached_tasks_names)}"
                )

            new_results = []
            if uncached_docs:
                uncached_tasks_names = {doc.task_name for doc in uncached_docs}
                logger.info(
                    f"Cache: Processing {len(uncached_docs)}/{len(docs)} {cache_type.name.lower()} samples for task {', '.join(uncached_tasks_names)}"
                )
                new_results = func(self, uncached_docs, *args, **kwargs)

                # Store new results in cache
                cache.store_samples(uncached_docs, new_results, task_names, cache_type)

            # Merge cached and new results in original order
            new_idx = 0

            for i, result in enumerate(results):
                if result is None:  # nothing cached
                    results[i] = new_results[new_idx]
                    new_idx += 1

            if any(r is None for r in results):
                raise ValueError("Problem while loading and aggregating items from cache.")

            return results

        return wrapper

    return decorator

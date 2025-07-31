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
from pathlib import Path
from typing import Callable, Union, Dict, List, Optional, Set, Tuple

from datasets import Dataset, load_dataset

from lighteval.models.utils import ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


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
        - {model_hash}/
          - {task_name}.parquet
      - predictions/
        - {model_hash}/
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
        self.tokenization_dir = self.cache_dir / "tokenization" / self.model_hash
        self.predictions_dir = self.cache_dir / "predictions" / self.model_hash
        
        self.tokenization_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory indices for fast lookup
        self._tokenization_indices: Dict[str, Set[str]] = {}  # task_name -> {sample_ids}
        self._prediction_indices: Dict[str, Set[str]] = {}    # task_name -> {sample_ids}
        
        # Load existing indices
        self._build_indices()

    def _build_indices(self, model_config: ModelConfig):
        """Build in-memory indices from existing cache files."""
        logger.info("Building cache indices...")

        model_hash = self.get_model_hash(model_config)
        
        # Build tokenization indices       
        for parquet_file in self.tokenization_dir.glob("*.parquet"):
            try:
                task_name = parquet_file.split(".parquet")[0] # todo: change to nice regex
                dataset = load_dataset("parquet", data_files=str(parquet_file), split="train")
                self._tokenization_indices[task_name] = {sample["sample_id"] for sample in dataset}
            except Exception as e:
                logger.warning(f"Error loading tokenization cache {parquet_file}: {e}")

        # Build tokenization indices       
        for parquet_file in self.predictions_dir.glob("*.parquet"):
            try:
                task_name = parquet_file.split(".parquet")[0] # todo: change to nice regex
                dataset = load_dataset("parquet", data_files=str(parquet_file), split="train")
                self._prediction_indices[task_name] = {sample["sample_id"] for sample in dataset}
            except Exception as e:
                logger.warning(f"Error loading tokenization cache {parquet_file}: {e}")


    def get_model_hash(self, model_config: ModelConfig) -> str:
        """Create a hash for model configuration."""
        config_str = json.dumps(asdict(model_config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def has_tokenization(self, doc: Doc, model_hash: str, task_name: str) -> bool:
        """Check if tokenization results exist for a sample."""
        try:
            return doc.id in self._tokenization_indices[task_name]
        except KeyError:
            return False

    def has_prediction(self, doc: Doc, model_hash: str, task_name: str) -> bool:
        """Check if prediction results exist for a sample."""
        try:
            return doc.id in self._predictions_indices[task_name]
        except KeyError:
            return False

    def get_tokenization(self, docs: List[Doc], model_hash: str, task_name: str) -> Tuple[List[Optional[dict]], List[Doc]]:
        """
        Get cached tokenization results for docs.
        
        Returns:
            Tuple of (cached_results, uncached_docs) where cached_results[i] is None 
            if docs[i] is not cached, and uncached_docs contains docs that need processing.
        """
        cache_file = self.tokenization_dir / model_hash / f"{task_name}.parquet"
        
        if not cache_file.exists():
            return [None] * len(docs), docs

        try:
            dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
            dataset_df = dataset.to_pandas().set_index("sample_id")
        except Exception as e:
            logger.warning(f"Error loading tokenization cache for {task_name}: {e}")
            return [None] * len(docs), docs

        cached_results = []
        uncached_docs = []
        
        for doc in docs:
            if doc.id in dataset_df.index:
                row = dataset_df.loc[doc.id]
                cached_results.append({
                    "input_tokens": row["input_tokens"],
                    "context_tokens": row.get("context_tokens"),
                    "continuation_tokens": row.get("continuation_tokens"),
                    "input_text": row["input_text"]
                })
            else:
                cached_results.append(None)
                uncached_docs.append(doc)

        return cached_results, uncached_docs

    def store_tokenization(self, docs: List[Doc], tokenization_results: List[dict], model_hash: str, task_name: str):
        """Store tokenization results for docs."""
        if not tokenization_results:
            return

        # Prepare data for dataset
        task_data = []
        for doc, result in zip(docs, tokenization_results):
            task_data.append({
                "sample_id": doc.id,
                "input_text": result["input_text"],
                "input_tokens": result["input_tokens"],
                "context_tokens": result.get("context_tokens"),
                "continuation_tokens": result.get("continuation_tokens"),
            })

        # Save dataset
        model_cache_dir = self.tokenization_dir / model_hash
        model_cache_dir.mkdir(exist_ok=True)
        cache_file = model_cache_dir / f"{task_name}.parquet"
        
        # Load existing data if present
        existing_data = []
        if cache_file.exists():
            try:
                existing_dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                existing_data = existing_dataset.to_list()
            except Exception as e:
                logger.warning(f"Error loading existing tokenization cache for {task_name}: {e}")

        # Merge with new data (new data overwrites existing)
        existing_ids = {row["sample_id"] for row in existing_data}
        all_data = existing_data + [row for row in task_data if row["sample_id"] not in existing_ids]
        
        # Save updated dataset
        dataset = Dataset.from_list(all_data)
        dataset.to_parquet(str(cache_file))
        
        # Update indices
        if model_hash not in self._tokenization_indices:
            self._tokenization_indices[model_hash] = set()
        
        for row in task_data:
            self._tokenization_indices[model_hash].add(row["sample_id"])

        logger.info(f"Stored tokenization results for {len(docs)} samples in {task_name}")

    def get_predictions(self, docs: List[Doc], model_hash: str, task_name: str) -> Tuple[List[Optional[ModelResponse]], List[Doc]]:
        """
        Get cached prediction results for docs.
        
        Returns:
            Tuple of (cached_responses, uncached_docs)
        """
        cache_file = self.predictions_dir / model_hash / f"{task_name}.parquet"
        
        if not cache_file.exists():
            return [None] * len(docs), docs

        try:
            dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
            dataset_df = dataset.to_pandas().set_index("sample_id")
        except Exception as e:
            logger.warning(f"Error loading prediction cache for {task_name}: {e}")
            return [None] * len(docs), docs

        cached_results = []
        uncached_docs = []
        
        for doc in docs:
            if doc.id in dataset_df.index:
                row = dataset_df.loc[doc.id]
                cached_results.append(ModelResponse(
                    input=row.get("input", ""),
                    text=json.loads(row["text"]) if isinstance(row["text"], str) else row["text"],
                    input_tokens=json.loads(row["input_tokens"]) if isinstance(row["input_tokens"], str) else row["input_tokens"],
                    output_tokens=json.loads(row["output_tokens"]) if isinstance(row["output_tokens"], str) else row["output_tokens"],
                    logprobs=json.loads(row["logprobs"]) if row.get("logprobs") and isinstance(row["logprobs"], str) else row.get("logprobs"),
                    argmax_logits_eq_gold=json.loads(row["argmax_logits_eq_gold"]) if row.get("argmax_logits_eq_gold") and isinstance(row["argmax_logits_eq_gold"], str) else row.get("argmax_logits_eq_gold"),
                ))
            else:
                cached_results.append(None)
                uncached_docs.append(doc)

        return cached_results, uncached_docs

    def store_predictions(self, docs: List[Doc], responses: List[ModelResponse], model_hash: str, task_name: str):
        """Store prediction results for docs."""
        if not responses:
            return

        # Prepare data for dataset
        task_data = []
        for doc, response in zip(docs, responses):
            task_data.append({
                "sample_id": doc.id,
                "input": response.input or "",
                "text": json.dumps(response.text) if response.text else "[]",
                "input_tokens": json.dumps(response.input_tokens) if response.input_tokens else "[]",
                "output_tokens": json.dumps(response.output_tokens) if response.output_tokens else "[]",
                "logprobs": json.dumps(response.logprobs) if response.logprobs else None,
                "argmax_logits_eq_gold": json.dumps(response.argmax_logits_eq_gold) if response.argmax_logits_eq_gold else None,
            })

        # Save dataset
        model_cache_dir = self.predictions_dir / model_hash
        model_cache_dir.mkdir(exist_ok=True)
        cache_file = model_cache_dir / f"{task_name}.parquet"
        
        # Load existing data if present
        existing_data = []
        if cache_file.exists():
            try:
                existing_dataset = load_dataset("parquet", data_files=str(cache_file), split="train")
                existing_data = existing_dataset.to_list()
            except Exception as e:
                logger.warning(f"Error loading existing prediction cache for {task_name}: {e}")

        # Merge with new data
        existing_ids = {row["sample_id"] for row in existing_data}
        all_data = existing_data + [row for row in task_data if row["sample_id"] not in existing_ids]
        
        # Save updated dataset
        dataset = Dataset.from_list(all_data)
        dataset.to_parquet(str(cache_file))
        
        # Update indices
        if model_hash not in self._prediction_indices:
            self._prediction_indices[model_hash] = set()
        
        for row in task_data:
            self._prediction_indices[model_hash].add(row["sample_id"])

        logger.info(f"Stored prediction results for {len(docs)} samples in {task_name}")


def cached(cache_type: str, extract_task_name: Optional[Callable] = None):
    """
    Decorator to cache method results based on Doc inputs.
    
    Args:
        cache_type: Type of cache ("tokenization" or "predictions")
        extract_task_name: Function to extract task name from method arguments.
                          If None, uses docs[0].task_name
    
    Usage:
        @cached("tokenization")
        def tok_encode_pair(self, docs: List[Doc], ...):
            # method implementation
            
        @cached("predictions")  
        def greedy_until(self, docs: List[Doc], ...):
            # method implementation
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, docs: Union[Doc, List[Doc]], *args, **kwargs):
            # Handle single doc or list of docs
            if isinstance(docs, Doc):
                docs = [docs]
            
            # Check if caching is enabled
            if not hasattr(self, '_cache') or self._cache is None:
                return func(self, docs, *args, **kwargs)
            
            cache: SampleCache = self._cache
            
            # Get model hash
            model_config = {
                "revision": getattr(self._config, "revision", "main"),
                "dtype": getattr(self._config, "dtype", ""),
                "add_special_tokens": getattr(self._config, "add_special_tokens", True),
                "pairwise_tokenization": getattr(self._config, "pairwise_tokenization", False),
                "max_model_length": getattr(self._config, "max_model_length", None),
            }
            model_hash = cache.get_model_hash(self._config.model_name, model_config)
            
            # Extract task name
            if extract_task_name:
                task_name = extract_task_name(self, docs, *args, **kwargs)
            else:
                task_name = docs[0].task_name if docs else "unknown"
            
            # Get cached results
            if cache_type == "tokenization":
                cached_results, uncached_docs = cache.get_tokenization(docs, model_hash, task_name)
            elif cache_type == "predictions":
                cached_results, uncached_docs = cache.get_predictions(docs, model_hash, task_name)
            else:
                raise ValueError(f"Unknown cache_type: {cache_type}")
            
            # Process uncached docs if any
            new_results = []
            if uncached_docs:
                logger.info(f"Cache miss: processing {len(uncached_docs)}/{len(docs)} samples for {cache_type}")
                new_results = func(self, uncached_docs, *args, **kwargs)
                
                # Store new results in cache
                if cache_type == "tokenization":
                    cache.store_tokenization(uncached_docs, new_results, model_hash, task_name)
                elif cache_type == "predictions":
                    cache.store_predictions(uncached_docs, new_results, model_hash, task_name)
            else:
                logger.info(f"Cache hit: all {len(docs)} samples found for {cache_type}")
            
            # Merge cached and new results in original order
            final_results = []
            new_idx = 0
            
            for i, doc in enumerate(docs):
                if cached_results[i] is not None:
                    final_results.append(cached_results[i])
                else:
                    final_results.append(new_results[new_idx])
                    new_idx += 1
            
            return final_results
            
        return wrapper
    return decorator
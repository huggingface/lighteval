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

import asyncio
import dataclasses
import functools
import hashlib
import json
import logging
from dataclasses import asdict
from typing import Callable, List

import diskcache

from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


def default_json_encoder(obj):
    """returns a string representation for objects not serializable by default json code"""
    if dataclasses.is_dataclass(obj):  # is dataclass instance
        return dataclasses.asdict(obj)
    elif hasattr(obj, "model_dump"):  # is pydantic BaseModel
        return obj.model_dump()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def hash_request(doc: Doc, **kwargs) -> str:
    """Create a hash for a request based on the doc and additional parameters."""
    return hashlib.sha256(
        json.dumps({"doc": doc, "kwargs": kwargs}, sort_keys=True, default=default_json_encoder).encode()
    ).hexdigest()


def cached(sampling_method: None | SamplingMethod = None):  # noqa: C901
    """
    Decorator to cache method results based on Doc inputs.

    Args:
        sampling_method: Sampling method to cache

    Usage:
        @cached(SamplingMethod.GENERATIVE)
        def greedy_until(self, docs: List[Doc], ...):
            # method implementation

    Returns:
        Callable: A decorator function that wraps the original function with caching functionality
    """

    def decorator(sampler: Callable):  # noqa: C901
        @functools.wraps(sampler)
        def sync_wrapper(self, docs: Doc | List[Doc], *args, **kwargs):
            if isinstance(docs, Doc):
                docs = [docs]

            results = [None] * len(docs)
            with diskcache.Cache(self.config.cache_dir) as cache:
                uncached_docs = []
                uncached_idxs = []

                for idx, doc in enumerate(docs):
                    key = hash_request(
                        doc, sampling_method=sampling_method, config=self.config, args=args, kwargs=kwargs
                    )
                    if key in cache:
                        logger.info("Cache hit")
                        results[idx] = cache[key]["response"]
                    else:
                        logger.info("Cache miss")
                        uncached_docs.append(doc)
                        uncached_idxs.append(idx)

                uncached_responses = sampler(self, uncached_docs, *args, **kwargs)

                for idx, doc, res in zip(uncached_idxs, uncached_docs, uncached_responses):
                    key = hash_request(
                        doc, sampling_method=sampling_method, config=self.config, args=args, kwargs=kwargs
                    )
                    cache[key] = {
                        "response": res,
                        "config": self.config.dict(),
                        "doc": asdict(doc),
                        "sampling_method": sampling_method,
                        "args": args,
                        "kwargs": kwargs,
                    }
                    results[idx] = res

            return results

        @functools.wraps(sampler)
        async def async_wrapper(self, docs: Doc | List[Doc], *args, **kwargs):
            if isinstance(docs, Doc):
                docs = [docs]

            results = [None] * len(docs)
            with diskcache.Cache(self.config.cache_dir) as cache:
                uncached_docs = []
                uncached_idxs = []

                for idx, doc in enumerate(docs):
                    key = hash_request(
                        doc, sampling_method=sampling_method, config=self.config, args=args, kwargs=kwargs
                    )
                    if key in cache:
                        logger.info("Cache hit")
                        results[idx] = cache[key]["response"]
                    else:
                        logger.info("Cache miss")
                        uncached_docs.append(doc)
                        uncached_idxs.append(idx)

                uncached_responses = await sampler(self, uncached_docs, *args, **kwargs)

                for idx, doc, res in zip(uncached_idxs, uncached_docs, uncached_responses):
                    key = hash_request(
                        doc, sampling_method=sampling_method, config=self.config, args=args, kwargs=kwargs
                    )
                    cache[key] = {
                        "response": res,
                        "config": self.config.dict(),
                        "doc": asdict(doc),
                        "sampling_method": sampling_method,
                        "args": args,
                        "kwargs": kwargs,
                    }
                    results[idx] = res

            return results

        return sync_wrapper if not asyncio.iscoroutinefunction(sampler) else async_wrapper

    return decorator

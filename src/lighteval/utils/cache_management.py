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
import functools
import hashlib
import json
import logging
from dataclasses import asdict
from typing import Callable, List

import diskcache

from lighteval.logging.evaluation_tracker import EnhancedJSONEncoder
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


def hash_request(doc: Doc, **kwargs) -> str:
    """Create a hash for a request based on the doc and additional parameters."""
    return hashlib.sha256(
        json.dumps({"doc": doc, "kwargs": kwargs}, sort_keys=True, cls=EnhancedJSONEncoder).encode()
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

    def decorator(model_call: Callable):  # noqa: C901
        @functools.wraps(model_call)
        def sync_wrapper(self, docs: List[Doc], *args, **kwargs):
            results = [None] * len(docs)
            with diskcache.Cache(self.config.cache_dir) as cache:
                uncached_docs = []
                uncached_idxs = []
                cache_hits, cache_misses = 0, 0

                for idx, doc in enumerate(docs):
                    key = hash_request(
                        doc, sampling_method=sampling_method, config=self.config, args=args, kwargs=kwargs
                    )
                    if key in cache:
                        cache_hits += 1
                        results[idx] = cache[key]["response"]
                    else:
                        cache_misses += 1
                        uncached_docs.append(doc)
                        uncached_idxs.append(idx)

                logger.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
                uncached_responses = model_call(self, uncached_docs, *args, **kwargs)

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

        @functools.wraps(model_call)
        async def async_wrapper(self, docs: List[Doc], *args, **kwargs):
            if isinstance(docs, Doc):
                docs = [docs]

            results = [None] * len(docs)
            with diskcache.Cache(self.config.cache_dir) as cache:
                uncached_docs = []
                uncached_idxs = []
                cache_hits, cache_misses = 0, 0

                for idx, doc in enumerate(docs):
                    key = hash_request(
                        doc, sampling_method=sampling_method, config=self.config, args=args, kwargs=kwargs
                    )
                    if key in cache:
                        cache_hits += 1
                        results[idx] = cache[key]["response"]
                    else:
                        cache_misses += 1
                        uncached_docs.append(doc)
                        uncached_idxs.append(idx)

                logger.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
                uncached_responses = await model_call(self, uncached_docs, *args, **kwargs)

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

        return sync_wrapper if not asyncio.iscoroutinefunction(model_call) else async_wrapper

    return decorator

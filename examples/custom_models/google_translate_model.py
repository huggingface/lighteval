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

import hashlib
import logging
import os
import time
from typing import Optional

import diskcache
import httpcore
import tenacity
from googletrans import Translator
from tqdm import tqdm
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)


logger = logging.getLogger(__name__)


class GoogleTranslateClient(LightevalModel):
    def __init__(self, config, env_config) -> None:
        self.model = config.model
        self.model_definition_file_path = config.model_definition_file_path

        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )

        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use a dummy tokenizer for compatibility

        # Needed to fix some googletrans bug
        # https://stackoverflow.com/questions/72796594/attributeerror-module-httpcore-has-no-attribute-synchttptransport#comment136664963_77334618
        setattr(httpcore, "SyncHTTPTransport", "AsyncHTTPProxy")

        self.translator = Translator()

        # Initialize disk cache
        cache_dir = os.path.join(os.getcwd(), ".translation_cache")
        self.cache = diskcache.Cache(cache_dir)

        self.max_retries = 3
        self.retry_delay = 1

    def _get_cache_key(self, context: str, src_lang: str, tgt_lang: str) -> str:
        """Generate a unique cache key for the translation request."""
        key_string = f"{context}|{src_lang}|{tgt_lang}"
        return hashlib.md5(key_string.encode()).hexdigest()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((Exception)),
        before_sleep=lambda retry_state: time.sleep(1),
    )
    def _translate_with_cache(self, context: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text using cache if available, otherwise call Google Translate with retry logic."""
        cache_key = self._get_cache_key(context, src_lang, tgt_lang)

        # Try to get from cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # If not in cache, translate and store
            translation = self.translator.translate(context, src=src_lang, dest=tgt_lang)
            result = translation.text
            self.cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Translation error: {str(e)}. Retrying...")
            # Re-initialize translator on error
            self.translator = Translator()
            raise  # Let tenacity handle the retry

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.
        Results are cached to disk to avoid repeated translations.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            for r in tqdm(dataset, desc="Batch", position=1, disable=False):
                # Extract source and target languages from task name
                # Format is like "community|sdst-text_level:de-fr|0"
                src_lang, tgt_lang = r.task_name.split("|")[1].split(":")[-1].split("-")

                context = r.context.replace(f"{src_lang.upper()}: ", "").replace(f"\n{tgt_lang.upper()}: ", "")
                result = self._translate_with_cache(context, src_lang, tgt_lang)

                cur_response = GenerativeResponse(
                    result=result,
                    logits=None,
                    generated_tokens=[],
                    input_tokens=[],
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    def tok_encode(self, text: str):
        return self.tokenizer.encode(text)

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

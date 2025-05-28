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

import diskcache
import tenacity
from deep_translator import GoogleTranslator
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
    def __init__(self, config) -> None:
        self.model = config.model_name
        self.model_definition_file_path = config.model_definition_file_path

        self.model_info = ModelInfo(
            model_name=config.model_name,
            model_sha="",
            model_dtype=None,
            model_size="",
        )

        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use a dummy tokenizer for compatibility

        # Deep-translator also supports other translators
        self.translator = GoogleTranslator()

        # Initialize disk cache
        cache_dir = os.path.join(os.getcwd(), ".translation_cache")
        self.cache = diskcache.Cache(cache_dir)

        self.max_retries = 3
        self.retry_delay = 1

    def _get_cache_key(self, context: str, src_lang: str, tgt_lang: str) -> str:
        """Generate a unique cache key for the translation request."""
        # IMPORTANT: In case we want to support other translators, we can add the translator name to the key
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
            result = self.cache[cache_key]
            if result is not None and result != "":
                return result
            logger.warning("Translation in cache is empty. Removing from cache and retrying...")
            del self.cache[cache_key]

        try:
            # Updated translation call for deep-translator
            self.translator.source = src_lang
            self.translator.target = tgt_lang
            result = self.translator.translate(context)
            if result is None or result == "":
                result = ""

            self.cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Translation error: {str(e)}. Retrying...")
            raise  # Let tenacity handle the retry

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
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

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            for r in tqdm(split, desc="Batch", position=1, disable=False):
                # Extract source and target languages from task name
                # Format is like "community|sdst-text_level:de-fr|0"
                src_lang, tgt_lang = r.task_name.split("|")[1].split(":")[-1].split("-")

                context = r.context.replace(f"{src_lang.upper()}: ", "").replace(f"\n{tgt_lang.upper()}: ", "")
                result = self._translate_with_cache(context, src_lang, tgt_lang)
                if result is None:
                    result = ""  # Set to empty string to prevent errors in metric computation

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
        return text

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(
        self,
        requests: list[LoglikelihoodRollingRequest],
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError

    def loglikelihood_single_token(
        self,
        requests: list[LoglikelihoodSingleTokenRequest],
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

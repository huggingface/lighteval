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
from typing import Optional

import pycountry
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    SeamlessM4Tv2ForTextToText,
)

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo, TokenSequence
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


class LocalMTClient(LightevalModel):
    """
    A custom model implementation for local machine translation models, specifically supporting:
    - SeamlessM4T v2 models from Meta
    - MADLAD-400 models from Google

    This class provides a unified interface for both model families while handling their different
    tokenization and generation approaches transparently.

    Args:
        config (CustomModelConfig): Configuration containing:
            - model (str): Model identifier/path (e.g. "facebook/seamless-m4t-v2-large" or "google/madlad400-7b-mt")
            - model_definition_file_path (str): Path to this model definition file
        env_config: Environment configuration (unused)

    The model automatically detects whether to load SeamlessM4T or MADLAD based on the model identifier
    and initializes the appropriate tokenizer and model.

    Translation tasks should specify the source and target languages in the format:
    "{task_name}|{...}:{src}-{tgt}"
    where src and tgt are ISO language codes (2 or 3 letter codes supported).

    Example:
        ```lighteval custom facebook/seamless-m4t-v2-large examples/custom_models/local_mt_model.py "lighteval|wmt20:fr-de|0|0" --max-samples 10 --save-details
        ```

    Note:
        - SeamlessM4T models use the AutoProcessor for tokenization
        - MADLAD models use the standard AutoTokenizer
        - Language codes are automatically converted to 3-letter ISO codes for SeamlessM4T
    """

    def __init__(self, config, env_config) -> None:
        self.model = config.model
        self.model_definition_file_path = config.model_definition_file_path
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size=-1,
        )

        # Update model initialization to handle both models
        if "seamless-m4t" in config.model:
            self._tokenizer = AutoProcessor.from_pretrained(config.model)
            self._model = SeamlessM4Tv2ForTextToText.from_pretrained(config.model)
            self.model_type = "seamless-4mt"
            self.batch_size = 1
            logger.info(
                "Using batch size of 1 for seamless-4mt model because it the target language needs to be set for the entire batch."
            )
        elif "madlad400" in config.model:
            self._tokenizer = AutoTokenizer.from_pretrained(config.model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(config.model)
            self.model_type = "madlad400"
        else:
            raise ValueError(f"Unsupported model: {config.model}")

        self._model.to(self.device)
        self._model.eval()

    def _convert_to_iso3(self, lang_code: str) -> str:
        """Convert 2-letter ISO code to 3-letter ISO code."""
        try:
            return pycountry.languages.get(alpha_2=lang_code.lower()).alpha_3
        except AttributeError:
            # If conversion fails, return the original code
            return lang_code

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

        def get_langs(task_name: str) -> tuple[str, str]:
            src, tgt = task_name.split("|")[1].split(":")[-1].split("-")
            if self.model_type == "seamless-4mt":
                return self._convert_to_iso3(src), self._convert_to_iso3(tgt)
            return src, tgt

        # Prepare all inputs first for creating the GenerativeTaskDataset
        prepared_requests = []
        for request in requests:
            src_lang, tgt_lang = get_langs(request.task_name)
            request.context = request.context.replace(f"{src_lang.upper()}: ", "").replace(
                f"\n{tgt_lang.upper()}: ", ""
            )
            if self.model_type == "madlad400":
                request.context = f"<2{tgt_lang}> {request.context}"

            request.tokenized_context = self.tok_encode(request.context)
            prepared_requests.append(request)

        # Create dataset after preparation
        dataset = GenerativeTaskDataset(requests=prepared_requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []
        batch_size = override_bs or self.batch_size

        for split_start, split_end in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,
        ):
            # Get all requests for this split directly from sorted_data
            current_requests = dataset.sorted_data[split_start:split_end]

            # Process in batches
            for batch_idx in tqdm(
                range(0, len(current_requests), batch_size), desc="Batches", position=1, disable=False
            ):
                batch = current_requests[batch_idx : batch_idx + batch_size]

                # Batch tokenize all inputs together instead of concatenating pre-tokenized inputs because of the padding
                batch_texts = [r.context for r in batch]

                # This is the tokenization step that really counts, as it actually gets used
                tokenizer_kwargs = {"text": batch_texts, "return_tensors": "pt", "padding": True}
                if self.model_type == "seamless-4mt":
                    src_lang = get_langs(batch[0].task_name)[0]
                    tokenizer_kwargs["src_lang"] = src_lang

                input_ids, attention_mask = self._tokenizer(**tokenizer_kwargs).to(self.device).values()

                generation_sizes = [r.generation_size for r in batch]
                assert set(generation_sizes) == {generation_sizes[0]}, "All generation sizes must be the same"

                # Use unpacked values directly
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": generation_sizes[0],
                }
                if self.model_type == "seamless-4mt":
                    tgt_lang = get_langs(batch[0].task_name)[1]
                    generate_kwargs["tgt_lang"] = tgt_lang

                output_ids = self._model.generate(**generate_kwargs)
                translations = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # Create responses for the batch
                for input_tokens, output_tokens, translation in zip(input_ids, output_ids, translations):
                    results.append(
                        GenerativeResponse(
                            input_tokens=input_tokens,
                            generated_tokens=output_tokens,
                            result=translation,
                            logits=None,
                        )
                    )

        return dataset.get_original_order(results)

    def cleanup(self):
        import gc

        logger.info("Cleaning up GPU memory for local MT client.")

        # Show GPU memory before cleanup
        if torch.cuda.is_available():
            logger.info(f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Delete model and move to CPU
        if hasattr(self, "_model"):
            self._model.cpu()
            del self._model
            self._model = None

        if hasattr(self, "_tokenizer"):
            del self._tokenizer
            self._tokenizer = None

        torch.cuda.empty_cache()
        gc.collect()

        # Show GPU memory after cleanup
        if torch.cuda.is_available():
            logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    @property
    def tokenizer(self):
        return self._tokenizer

    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None) -> TokenSequence:
        return self._tokenizer(text=str_to_encode, add_special_tokens=add_special_tokens or False).to(self.device)

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

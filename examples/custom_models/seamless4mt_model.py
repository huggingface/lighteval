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
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

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


class Seamless4MTClient(LightevalModel):
    def __init__(self, config, env_config) -> None:
        self.model = config.model
        self.model_definition_file_path = config.model_definition_file_path

        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )
        self._tokenizer = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self._model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")

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
            return self._convert_to_iso3(src), self._convert_to_iso3(tgt)

        # Prepare all inputs first
        prepared_requests = []
        for request in requests:
            src_lang, tgt_lang = get_langs(request.task_name)
            request.context = request.context.replace(f"{src_lang.upper()}: ", "").replace(
                f"\n{tgt_lang.upper()}: ", ""
            )
            request.tokenized_context = self._tokenizer(
                text=request.context, src_lang=src_lang, return_tensors="pt", padding=True
            )
            prepared_requests.append(request)

        # Create dataset after preparation
        dataset = GenerativeTaskDataset(requests=prepared_requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []
        batch_size = override_bs or 32

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
            for batch_idx in range(0, len(current_requests), batch_size):
                batch = current_requests[batch_idx : batch_idx + batch_size]

                # Batch tokenize all inputs together instead of concatenating pre-tokenized inputs
                batch_texts = [r.context for r in batch]
                src_lang = get_langs(batch[0].task_name)[0]  # All source languages should be the same in a batch

                # Unpack the tokenizer output into input_ids and attention_mask
                input_ids, attention_mask = self._tokenizer(
                    text=batch_texts, src_lang=src_lang, return_tensors="pt", padding=True
                ).values()

                tgt_langs = [get_langs(r.task_name)[1] for r in batch]
                assert set(tgt_langs) == {tgt_langs[0]}, "All target languages must be the same"

                # Use unpacked values directly
                output_ids = self._model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, tgt_lang=tgt_langs[0]
                )
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

    @property
    def tokenizer(self):
        return self._tokenizer

    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None) -> TokenSequence:
        return self._tokenizer(
            text=str_to_encode, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens or False
        )

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

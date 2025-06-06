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
import shutil
from contextlib import nullcontext

import torch
import transformers
from transformers import AutoModelForCausalLM

from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.models.utils import _get_dtype
from lighteval.utils.imports import NO_PEFT_ERROR_MSG, is_peft_available


logger = logging.getLogger(__name__)

if is_peft_available():
    from peft import PeftModel


class AdapterModelConfig(TransformersModelConfig):
    # Adapter models have the specificity that they look at the base model (= the parent) for the tokenizer and config
    base_model: str
    adapter_weights: bool

    def model_post_init(self, __context):
        if not is_peft_available():
            raise ImportError(NO_PEFT_ERROR_MSG)


class AdapterModel(TransformersModel):
    def _create_auto_model(self) -> transformers.PreTrainedModel:
        """Returns a PeftModel from a base model and a version fined tuned using PEFT."""
        torch_dtype = _get_dtype(self.config.dtype)
        model_parallel, max_memory, device_map = self.init_model_parallel(self.config.model_parallel)
        self.config.model_parallel = model_parallel

        adapter_weights = self.config.pretrained
        merged_path = f"{adapter_weights}-adapter-applied"

        if self.config.dtype == "4bit":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif self.config.dtype == "8bit":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        if self.accelerator.is_local_main_process if self.accelerator is not None else nullcontext():
            logger.info(f"Loading model from {adapter_weights} and applying adapter to {self.config.base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                self.config.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            # resize model for adapters with added tokens
            token_diff = len(self._tokenizer) - base.config.vocab_size
            if token_diff != 0:
                if token_diff > 0:
                    logger.info(
                        f"You're using the adapter model's tokenizer, which has more tokens than the base model. Adding {token_diff} token(s)."
                    )
                else:
                    logger.info(
                        f"You're using the adapter model's tokenizer, which has fewer tokens than the base model. Removing {abs(token_diff)} token(s)."
                    )
                base.resize_token_embeddings(len(self._tokenizer))
            # Should pass revision
            model = PeftModel.from_pretrained(base, adapter_weights)
            model = model.merge_and_unload()

            logger.info("Saving model with adapter applied")
            base.save_pretrained(merged_path)

        logger.info(f"Loading model from {merged_path}")

        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            max_memory=max_memory,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            quantization_config=quantization_config,
        )

        return model

    def cleanup(self):
        try:
            tmp_weights_dir = f"{self.model_name}-adapter-applied"
            shutil.rmtree(tmp_weights_dir)
            logger.info(f"Removed {tmp_weights_dir}")
        except OSError:
            pass

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
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.models.utils import _get_dtype, _get_model_sha


logger = logging.getLogger(__name__)


class DeltaModelConfig(TransformersModelConfig):
    """
    Configuration class for delta models (weight difference models).

    This configuration is used to load models that represent the difference between a
    fine-tuned model and its base model. The delta weights are added to the base model
    during loading to reconstruct the full fine-tuned model.

    Attributes:
        base_model (str):
            HuggingFace Hub model ID or path to the base model. This is the original
            pre-trained model that the delta was computed from.
        delta_weights (bool):
            Flag indicating that this is a delta model. Must be set to True.
    """

    # Delta models look at the pretrained (= the delta weights) for the tokenizer and model config
    base_model: str
    delta_weights: bool

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.model_name, revision="main")


class DeltaModel(TransformersModel):
    def _create_auto_model(
        self,
        config: DeltaModelConfig,
    ) -> AutoModelForCausalLM:
        """Returns a model created by adding the weights of a delta model to a base model."""
        config.model_parallel, max_memory, device_map = self.init_model_parallel(config.model_parallel)
        torch_dtype = _get_dtype(config.dtype, self._config)

        delta_model = config.pretrained

        merged_path = f"{delta_model}-delta-applied"

        if self.accelerator.is_main_process if self.accelerator is not None else nullcontext():
            logger.info(f"Loading base and delta models from {config.base_model} and {delta_model}")
            base = AutoModelForCausalLM.from_pretrained(
                config.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            delta = AutoModelForCausalLM.from_pretrained(
                delta_model,
                revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
                assert name in delta.state_dict()
                param.data += delta.state_dict()[name]

            logger.info("Saving delta-applied model")
            base.save_pretrained(merged_path)

        logger.info(f"Loading delta-applied model from {delta_model}-delta-applied")

        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            max_memory=max_memory,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            quantization_config=config.quantization_config,
        )

        return model

    def cleanup(self):
        try:
            tmp_weights_dir = f"{self.model_name}-delta-applied"
            shutil.rmtree(tmp_weights_dir)
            logger.info(f"Removed {tmp_weights_dir}")
        except OSError:
            pass

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
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.models.utils import _get_dtype, _get_model_sha
from lighteval.utils.utils import EnvConfig


logger = logging.getLogger(__name__)


@dataclass
class DeltaModelConfig(TransformersModelConfig):
    # Delta models look at the pretrained (= the delta weights) for the tokenizer and model config
    base_model: str = None

    def __post_init__(self):
        self.revision = "main"

        if not self.base_model:  # must have a default value bc of dataclass inheritance, but can't actually be None
            raise ValueError("The base_model argument must not be null for a delta model config")

        return super().__post_init__()

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.pretrained, revision="main")


class DeltaModel(TransformersModel):
    def _create_auto_model(
        self,
        config: DeltaModelConfig,
        env_config: EnvConfig,
    ) -> AutoModelForCausalLM:
        """Returns a model created by adding the weights of a delta model to a base model."""
        config.model_parallel, max_memory, device_map = self.init_model_parallel(config.model_parallel)
        torch_dtype = _get_dtype(config.dtype, self._config)

        delta_model = config.pretrained

        merged_path = f"{delta_model}-delta-applied"

        if self.accelerator.is_main_process if self.accelerator is not None else nullcontext():
            logger.info(f"Loading base and delta models from {config.base_model} and {delta_model}")
            base = AutoModelForCausalLM.from_pretrained(
                config.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=env_config.token
            )
            delta = AutoModelForCausalLM.from_pretrained(
                delta_model,
                revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                token=env_config.token,
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
            cache_dir=env_config.cache_dir,
            quantization_config=config.quantization_config,
            token=env_config.token,
        )

        return model

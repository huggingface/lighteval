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

from contextlib import nullcontext

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import DeltaModelConfig
from lighteval.models.utils import _get_dtype
from lighteval.utils.utils import EnvConfig


class DeltaModel(BaseModel):
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
            hlog(f"Loading base and delta models from {config.base_model} and {delta_model}")
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

            hlog("Saving delta-applied model")
            base.save_pretrained(merged_path)

        hlog(f"Loading delta-applied model from {delta_model}-delta-applied")

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

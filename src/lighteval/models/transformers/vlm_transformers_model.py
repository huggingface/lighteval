# MIT License

# Copyright (c) 2025 The HuggingFace Team

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
import os
from datetime import timedelta
from typing import Optional, Tuple, Union

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object, get_max_memory
from pydantic import PositiveInt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig, _get_dtype, _get_model_sha, _simplify_name
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import (
    is_accelerate_available,
)


logger = logging.getLogger(__name__)


class BatchCollator:
    """Collator for batching requests"""

    def __init__(self, prompt_manager, processor, **kwargs):
        self.processor = processor
        self.prompt_manager = prompt_manager
        self.kwargs = kwargs

    def __call__(self, requests: list[Doc]) -> Tuple[dict[str, torch.Tensor], list[Doc], list[str]]:
        texts = [self.prompt_manager.prepare_prompt_multimodal(request) for request in requests]
        images = [request.images for request in requests]
        inputs = self.processor(text=texts, images=images, **self.kwargs)
        return inputs, requests, texts


class VLMTransformersModelConfig(ModelConfig):
    """
    Configuration class for VLM (image-text-to-text) models.

    Attributes:
        model_name (str):
            HuggingFace Hub model ID name or the path to a pre-trained
            model to load. This is effectively the `pretrained_model_name_or_path`
            argument of `from_pretrained` in the HuggingFace `transformers` API.
        processor (Optional[str]): HuggingFace Hub processor ID that will be
            used for preprocessing images and text.
        use_fast_image_processor (Optional[bool]):
            Whether to use a fast image processor. Not all VLMs support this yet.
        subfolder (Optional[str]): The subfolder within the model repository.
        revision (str): The revision of the model.
        batch_size (int): The batch size for model training.
        generation_size (Optional[int]): The maximum number of tokens to generate.
        max_length (Optional[int]): The maximum length of the input + generated output.
        add_special_tokens (bool, optional, defaults to True): Whether to add special tokens to the input sequences.
        model_parallel (bool, optional, defaults to None):
            True/False: force to use or not the `accelerate` library to load a large
            model across multiple devices.
            Default: None which corresponds to comparing the number of processes with
                the number of GPUs. If it's smaller => model-parallelism, else not.
        dtype (Union[str, torch.dtype], optional, defaults to None):
            Converts the model weights to `dtype`, if specified. Strings get
            converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
            Use `dtype="auto"` to derive the type from the model's weights.
        device (Union[int, str]): device to use for model training.
        quantization_config (Optional[BitsAndBytesConfig]): quantization
            configuration for the model, manually provided to load a normally floating point
            model at a quantized precision. Needed for 4-bit and 8-bit precision.
        trust_remote_code (bool): Whether to trust remote code during model
            loading.
    """

    model_name: str
    processor: str | None = None
    use_fast_image_processor: bool | None = None
    subfolder: str | None = None
    revision: str = "main"
    batch_size: PositiveInt = 1
    generation_size: PositiveInt | None = None
    max_length: PositiveInt | None = None
    add_special_tokens: bool = True
    model_parallel: bool | None = None
    dtype: str | None = None
    device: Union[int, str] = "cuda"
    trust_remote_code: bool = False
    use_chat_template: bool = False
    compile: bool = False
    device_map: str | None = None

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.model_name, revision=self.revision)

    def get_transformers_config(self) -> PretrainedConfig:
        revision = f"{self.revision}/{self.subfolder}" if self.subfolder else self.revision
        config = AutoConfig.from_pretrained(
            self.model_name,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
        )
        return config


class VLMTransformersModel(LightevalModel):
    def __init__(
        self,
        config: VLMTransformersModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""

        self.accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
        self.device = self.accelerator.device
        self.torch_dtype = _get_dtype(config.dtype)

        # Config attributes
        self.config = config
        self.use_chat_template = config.use_chat_template
        self.batch_size = config.batch_size

        # Model, config, and processor
        self.model_sha = config.get_model_sha()
        self.model_name = _simplify_name(config.model_name)
        self.model = self._create_auto_model()
        self.processor = self._create_auto_processor()
        self.transformers_config = config.get_transformers_config()

        # Attributes exposed by @property
        self._max_length = self._init_max_length()
        self._add_special_tokens = config.add_special_tokens or False

        # Generation config
        self.generation_config_dict = config.generation_parameters.to_transformers_dict()
        self.generation_config_dict["pad_token_id"] = self.pad_token_id
        self.generation_config_dict["eos_token_id"] = self.eos_token_id
        self.generation_config_dict["renormalize_logits"] = True

        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

        self.model_info = ModelInfo(
            model_name=self.config.model_name,
            model_sha=self.model_sha,
            model_dtype=config.dtype,
        )

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self):
        return self._max_length

    @property
    def disable_tqdm(self) -> bool:
        disable_tqdm = False
        if self.accelerator:
            disable_tqdm = bool(not self.accelerator.is_main_process)
        return disable_tqdm

    # Copied from ./transformers_model.py
    def init_model_parallel(self, model_parallel: bool | None = None) -> Tuple[bool, Optional[dict], Optional[str]]:
        """Compute all the parameters related to model_parallel"""
        if not is_accelerate_available():
            return False, None, None

        self.num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.num_machines = torch.cuda.device_count() // self.num_local_processes

        if self.num_machines == 1:
            logger.info("We are not in a distributed setting. Setting model_parallel to False.")
            model_parallel = False

        if model_parallel is None:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            model_parallel = bool(self.num_local_processes < len(max_memory_all_gpus))
            logger.info(
                f"Setting model parallel to {model_parallel} since "
                f"the number of local processes is {self.num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )
        if model_parallel is True:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_mem_this_process = {
                k: v
                for k, v in max_memory_all_gpus.items()
                if k % self.num_local_processes == (self.accelerator.process_index % self.num_local_processes)
            }
            device_map = "auto"
            logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_mem_this_process} and device map to {device_map}"
            )
        else:
            max_mem_this_process = None
            device_map = None
            logger.info(
                f"Model parallel was set to False, max memory set to {max_mem_this_process} and device map to {device_map}"
            )
        return model_parallel, max_mem_this_process, device_map

    @staticmethod
    def _get_quantization_config(config: VLMTransformersModelConfig) -> BitsAndBytesConfig | None:
        if config.dtype == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif config.dtype == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        return quantization_config

    def _create_auto_model(self):
        model_parallel, max_memory, device_map = self.init_model_parallel(self.config.model_parallel)
        self.config.model_parallel = model_parallel

        quantization_config = self._get_quantization_config(self.config)

        subfolder = self.config.subfolder
        revision = f"{self.config.revision}/{subfolder}" if subfolder is not None else self.config.revision

        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name,
            revision=revision,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=self.torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=self.config.trust_remote_code,
        )
        model.eval()
        torch.set_grad_enabled(False)

        if self.config.compile:
            raise NotImplementedError("Compiling VLM models is not supported yet")

        # We are in DP (and launch the script with `accelerate launch`)
        if model_parallel is False and self.config.dtype not in ["4bit", "8bit"]:
            logger.info(f"Using Data Parallelism, putting model on device {self.device}")
            model = model.to(self.device)

        return model

    def _create_auto_processor(self):
        """
        Create a transformers `Processor` for VLM (image-text-to-text) model.

        Returns:
            transformers.ProcessorMixin: The created processor.
        """
        processor_name = self.config.processor or self.config.model_name
        revision, subfolder = self.config.revision, self.config.subfolder
        revision = revision if not subfolder else f"{revision}/{subfolder}"

        processor = AutoProcessor.from_pretrained(
            processor_name,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.config.trust_remote_code,
            use_fast=self.config.use_fast_image_processor,
        )

        return processor

    def _init_max_length(self) -> int:
        """Return the maximum sequence length of the model.

        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.

        Returns:
            int: Max length to use depending on the available args and config
        """
        if self.config.max_length is not None:
            return self.config.max_length

        # Try to get the sequence length from the model config. It's no super robust
        text_model_config = self.transformers_config.get_text_config()
        max_seq_length = getattr(text_model_config, "max_position_embeddings", None)
        if max_seq_length is not None:
            return max_seq_length

        logger.warning(
            "No max_length attribute found in the model config. Using the default max sequence length setting `2048`. "
            "It is recommended to set max_length trough the model args: max_length=..."
        )

        return 2048

    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """

        # Tokenizing context for sorting in the dataset

        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        collator = BatchCollator(
            self.prompt_manager,
            self.processor,
            truncation="longest_first",  # we truncate to the model max length if needed
            padding="longest",  # we pad to the longest sequence
            max_length=self.max_length - 1,  # we should always allow minimum one token of generation
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

        results = []
        for split in dataset.splits_iterator():
            batch_size = self.batch_size or 1
            dataloader = DataLoader(split, batch_size=batch_size, collate_fn=collator)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch_inputs, batch_requests, input_context in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=True, disable=self.disable_tqdm
            ):
                batch_inputs = batch_inputs.to(self.device)
                if self.torch_dtype is not None:
                    batch_inputs = batch_inputs.to(self.torch_dtype)

                max_new_tokens = self.config.generation_size or batch_requests[0].generation_size
                num_samples = batch_requests[0].num_samples
                do_sample = num_samples > 1 or self.generation_config_dict["temperature"] > 0

                if num_samples > 1 and self.generation_config_dict["temperature"] == 0:
                    raise ValueError(
                        "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                    )

                outputs = self.model.generate(
                    **batch_inputs,
                    **self.generation_config_dict,  # custom generation params
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=batch_requests[0].num_samples,
                    output_logits=batch_requests[0].use_logits,
                    do_sample=do_sample,
                )
                input_tokens = batch_inputs.input_ids
                generated_tokens = outputs.sequences[:, input_tokens.shape[1] :]
                generated_texts = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
                attention_mask = batch_inputs["attention_mask"]
                padded_tokens_count = (attention_mask == 0).sum(dim=1)

                batch_results = []
                for i in range(len(generated_texts)):
                    generated_response = ModelResponse(
                        input=input_context[i],
                        text=generated_texts[i],
                        output_tokens=generated_tokens[i].cpu().numpy(),
                        input_tokens=input_tokens[i].cpu().numpy(),
                        truncated_tokens_count=-1,
                        padded_tokens_count=padded_tokens_count[i].item(),
                        logits=outputs.logits[i].cpu().numpy() if outputs.logits is not None else None,
                    )
                    batch_results.append(generated_response)

                if self.accelerator:
                    batch_results = gather_object(batch_results)

                results.extend(batch_results)

        return dataset.get_original_order(results)

    def loglikelihood(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        raise NotImplementedError()

    def loglikelihood_rolling(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        raise NotImplementedError()

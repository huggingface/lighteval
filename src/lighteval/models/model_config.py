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

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig, BitsAndBytesConfig, GPTQConfig, PretrainedConfig

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.utils import _get_model_sha
from lighteval.utils.imports import (
    NO_AUTOGPTQ_ERROR_MSG,
    NO_BNB_ERROR_MSG,
    NO_PEFT_ERROR_MSG,
    is_accelerate_available,
    is_autogptq_available,
    is_bnb_available,
    is_peft_available,
)
from lighteval.utils.utils import EnvConfig, boolstring_to_bool


if is_accelerate_available():
    from accelerate import Accelerator


@dataclass
class BaseModelConfig:
    """
    Base configuration class for models.

    Attributes:
        pretrained (str):
            HuggingFace Hub model ID name or the path to a pre-trained
            model to load. This is effectively the `pretrained_model_name_or_path`
            argument of `from_pretrained` in the HuggingFace `transformers` API.
        accelerator (Accelerator): accelerator to use for model training.
        tokenizer (Optional[str]): HuggingFace Hub tokenizer ID that will be
            used for tokenization.
        multichoice_continuations_start_space (Optional[bool]): Whether to add a
            space at the start of each continuation in multichoice generation.
            For example, context: "What is the capital of France?" and choices: "Paris", "London".
            Will be tokenized as: "What is the capital of France? Paris" and "What is the capital of France? London".
            True adds a space, False strips a space, None does nothing
        pairwise_tokenization (bool): Whether to tokenize the context and continuation as separately or together.
        subfolder (Optional[str]): The subfolder within the model repository.
        revision (str): The revision of the model.
        batch_size (int): The batch size for model training.
        max_gen_toks (Optional[int]): The maximum number of tokens to generate.
        max_length (Optional[int]): The maximum length of the generated output.
        add_special_tokens (bool, optional, defaults to True): Whether to add special tokens to the input sequences.
           If `None`, the default value will be set to `True` for seq2seq models (e.g. T5) and
            `False` for causal models.
        model_parallel (bool, optional, defaults to False):
            True/False: force to use or not the `accelerate` library to load a large
            model across multiple devices.
            Default: None which corresponds to comparing the number of processes with
                the number of GPUs. If it's smaller => model-parallelism, else not.
        dtype (Union[str, torch.dtype], optional, defaults to None):):
            Converts the model weights to `dtype`, if specified. Strings get
            converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
            Use `dtype="auto"` to derive the type from the model's weights.
        device (Union[int, str]): device to use for model training.
        quantization_config (Optional[BitsAndBytesConfig]): quantization
            configuration for the model, manually provided to load a normally floating point
            model at a quantized precision. Needed for 4-bit and 8-bit precision.
        trust_remote_code (bool): Whether to trust remote code during model
            loading.

    Methods:
        __post_init__(): Performs post-initialization checks on the configuration.
        _init_configs(model_name, env_config): Initializes the model configuration.
        init_configs(env_config): Initializes the model configuration using the environment configuration.
        get_model_sha(): Retrieves the SHA of the model.

    """

    pretrained: str
    accelerator: "Accelerator" = None
    tokenizer: Optional[str] = None
    multichoice_continuations_start_space: Optional[bool] = None
    pairwise_tokenization: bool = False
    subfolder: Optional[str] = None
    revision: str = "main"
    batch_size: int = -1
    max_gen_toks: Optional[int] = 256
    max_length: Optional[int] = None
    add_special_tokens: bool = True
    model_parallel: Optional[bool] = None
    dtype: Optional[Union[str, torch.dtype]] = None
    device: Union[int, str] = "cuda"
    quantization_config: Optional[BitsAndBytesConfig] = None
    trust_remote_code: bool = False
    use_chat_template: bool = False
    compile: bool = False

    def __post_init__(self):
        # Making sure this parameter is a boolean
        self.multichoice_continuations_start_space = boolstring_to_bool(self.multichoice_continuations_start_space)

        if self.multichoice_continuations_start_space is not None:
            if self.multichoice_continuations_start_space:
                hlog(
                    "You set `multichoice_continuations_start_space` to true. This will force multichoice continuations to use a starting space"
                )
            else:
                hlog(
                    "You set `multichoice_continuations_start_space` to false. This will remove a leading space from multichoice continuations, if present."
                )

        self.model_parallel = boolstring_to_bool(self.model_parallel)
        self.compile = boolstring_to_bool(self.compile)

        if self.quantization_config is not None and not is_bnb_available():
            raise ImportError(NO_BNB_ERROR_MSG)

        if not isinstance(self.pretrained, str):
            raise ValueError("Pretrained model name must be passed as string.")
        if not isinstance(self.device, str):
            raise ValueError("Current device must be passed as string.")

    def _init_configs(self, model_name: str, env_config: EnvConfig) -> PretrainedConfig:
        revision = self.revision
        if self.subfolder:
            revision = f"{self.revision}/{self.subfolder}"
        auto_config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
            cache_dir=env_config.cache_dir,
            token=env_config.token,
        )

        # Gathering the model's automatic quantization config, if available
        try:
            model_auto_quantization_config = auto_config.quantization_config
            hlog("An automatic quantization config was found in the model's config. Using it to load the model")
        except (AttributeError, KeyError):
            model_auto_quantization_config = None

        if model_auto_quantization_config is not None:
            if self.quantization_config is not None:
                # We don't load models quantized by default with a different user provided conf
                raise ValueError("You manually requested quantization on a model already quantized!")

            # We add the quantization to the model params we store
            if model_auto_quantization_config["quant_method"] == "gptq":
                if not is_autogptq_available():
                    raise ImportError(NO_AUTOGPTQ_ERROR_MSG)
                auto_config.quantization_config["use_exllama"] = None
                self.quantization_config = GPTQConfig(**auto_config.quantization_config, disable_exllama=True)
            elif model_auto_quantization_config["quant_method"] == "bitsandbytes":
                if not is_bnb_available():
                    raise ImportError(NO_BNB_ERROR_MSG)
                self.quantization_config = BitsAndBytesConfig(**auto_config.quantization_config)

        return auto_config

    def init_configs(self, env_config: EnvConfig) -> PretrainedConfig:
        return self._init_configs(self.pretrained, env_config=env_config)

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.pretrained, revision=self.revision)


@dataclass
class DeltaModelConfig(BaseModelConfig):
    # Delta models look at the pretrained (= the delta weights) for the tokenizer and model config
    base_model: str = None

    def __post_init__(self):
        self.revision = "main"

        if not self.base_model:  # must have a default value bc of dataclass inheritance, but can't actually be None
            raise ValueError("The base_model argument must not be null for a delta model config")

        return super().__post_init__()

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.pretrained, revision="main")


@dataclass
class AdapterModelConfig(BaseModelConfig):
    # Adapter models have the specificity that they look at the base model (= the parent) for the tokenizer and config
    base_model: str = None

    def __post_init__(self):
        if not is_peft_available():
            raise ImportError(NO_PEFT_ERROR_MSG)

        if not self.base_model:  # must have a default value bc of dataclass inheritance, but can't actually be None
            raise ValueError("The base_model argument must not be null for an adapter model config")

        return super().__post_init__()

    def init_configs(self, env_config: EnvConfig):
        return self._init_configs(self.base_model, env_config)


@dataclass
class VLLMModelConfig:
    pretrained: str
    gpu_memory_utilisation: float = 0.9  # lower this if you are running out of memory
    revision: str = "main"  # revision of the model
    dtype: str | None = None
    tensor_parallel_size: int = 1  # how many GPUs to use for tensor parallelism
    pipeline_parallel_size: int = 1  # how many GPUs to use for pipeline parallelism
    data_parallel_size: int = 1  # how many GPUs to use for data parallelism
    max_model_length: int | None = None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    swap_space: int = 4  # CPU swap space size (GiB) per GPU.
    seed: int = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = (
        True  # whether to add a space at the start of each continuation in multichoice generation
    )
    pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.

    subfolder: Optional[str] = None
    temperature: float = 0.6  # will be used for multi sampling tasks, for tasks requiring no sampling, this will be ignored and set to 0.


@dataclass
class OpenAIModelConfig:
    model: str


@dataclass
class TGIModelConfig:
    inference_server_address: str
    inference_server_auth: str
    model_id: str


@dataclass
class DummyModelConfig:
    seed: int = 42


@dataclass
class InferenceModelConfig:
    model: str
    add_special_tokens: bool = True


@dataclass
class InferenceEndpointModelConfig:
    name: str
    repository: str
    accelerator: str
    vendor: str
    region: str
    instance_size: str
    instance_type: str
    model_dtype: str
    framework: str = "pytorch"
    endpoint_type: str = "protected"
    should_reuse_existing: bool = False
    add_special_tokens: bool = True
    revision: str = "main"
    namespace: str = None  # The namespace under which to launch the endopint. Defaults to the current user's namespace
    image_url: str = None
    env_vars: dict = None

    def get_dtype_args(self) -> Dict[str, str]:
        model_dtype = self.model_dtype.lower()
        if model_dtype in ["awq", "eetq", "gptq"]:
            return {"QUANTIZE": model_dtype}
        if model_dtype == "8bit":
            return {"QUANTIZE": "bitsandbytes"}
        if model_dtype == "4bit":
            return {"QUANTIZE": "bitsandbytes-nf4"}
        if model_dtype in ["bfloat16", "float16"]:
            return {"DTYPE": model_dtype}
        return {}

    def get_custom_env_vars(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.env_vars.items()} if self.env_vars else {}

    @staticmethod
    def nullable_keys() -> list[str]:
        """
        Returns the list of optional keys in an endpoint model configuration. By default, the code requires that all the
        keys be specified in the configuration in order to launch the endpoint. This function returns the list of keys
        that are not required and can remain None.
        """
        return ["namespace", "env_vars", "image_url"]

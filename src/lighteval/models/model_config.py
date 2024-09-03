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
import yaml
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
    gpu_memory_utilisation: float = 0.8
    batch_size: int = -1
    revision: str = "main"
    dtype: str | None = None
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    max_model_length: int = 1024
    swap_space: int = 4  # CPU swap space size (GiB) per GPU.
    seed: int = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = True
    subfolder: Optional[str] = None


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


def create_model_config(  # noqa: C901
    use_chat_template: bool,
    override_batch_size: int,
    accelerator: Union["Accelerator", None],
    model_args: Union[str, dict] = None,
    model_config_path: str = None,
) -> Union[
    BaseModelConfig,
    AdapterModelConfig,
    DeltaModelConfig,
    TGIModelConfig,
    InferenceEndpointModelConfig,
    DummyModelConfig,
    VLLMModelConfig,
]:
    """
    Create a model configuration based on the provided arguments.

    Args:
        accelerator(Union[Accelerator, None]): accelerator to use for model training.
        use_chat_template (bool): whether to use the chat template or not. Set to True for chat or ift models
        override_batch_size (int): frozen batch size to use
        model_args (Optional[Union[str, dict]]): Parameters to create the model, passed as a string (like the CLI kwargs or dict).
            This option only allows to create a dummy model using `dummy` or a base model (using accelerate or no accelerator), in
            which case corresponding full model args available are the arguments of the [[BaseModelConfig]].
            Minimal configuration is `pretrained=<name_of_the_model_on_the_hub>`.
        model_config_path (Optional[str]): Path to the parameters to create the model, passed as a config file. This allows to create
            all possible model configurations (base, adapter, peft, inference endpoints, tgi...)

    Returns:
        Union[BaseModelConfig, AdapterModelConfig, DeltaModelConfig, TGIModelConfig, InferenceEndpointModelConfig, DummyModelConfig]: model configuration.

    Raises:
        ValueError: If both an inference server address and model arguments are provided.
     ValueError: If multichoice continuations both should start with a space and should not start with a space.
        ValueError: If a base model is not specified when using delta weights or adapter weights.
        ValueError: If a base model is specified when not using delta weights or adapter weights.
    """
    if model_args is None and model_config_path is None:
        raise ValueError("You can't create a model without either a list of model_args or a model_config_path.")

    if model_args:
        if isinstance(model_args, str):
            model_args = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in model_args.split(",")}

        if model_args.pop("dummy", False):
            return DummyModelConfig(**model_args)

        if model_args.pop("vllm", False):
            return VLLMModelConfig(**model_args)

        model_args["accelerator"] = accelerator
        model_args["use_chat_template"] = use_chat_template
        model_args["compile"] = bool(model_args["compile"]) if "compile" in model_args else False

        return BaseModelConfig(**model_args)

    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)["model"]

    if config["type"] == "tgi":
        return TGIModelConfig(
            inference_server_address=config["instance"]["inference_server_address"],
            inference_server_auth=config["instance"]["inference_server_auth"],
            model_id=config["instance"]["model_id"],
        )

    if config["type"] == "endpoint":
        reuse_existing_endpoint = config["base_params"].get("reuse_existing", None)
        complete_config_endpoint = all(
            val not in [None, ""]
            for key, val in config.get("instance", {}).items()
            if key not in InferenceEndpointModelConfig.nullable_keys()
        )
        if reuse_existing_endpoint or complete_config_endpoint:
            return InferenceEndpointModelConfig(
                name=config["base_params"]["endpoint_name"].replace(".", "-").lower(),
                repository=config["base_params"]["model"],
                model_dtype=config["base_params"]["dtype"],
                revision=config["base_params"]["revision"] or "main",
                should_reuse_existing=reuse_existing_endpoint,
                accelerator=config["instance"]["accelerator"],
                region=config["instance"]["region"],
                vendor=config["instance"]["vendor"],
                instance_size=config["instance"]["instance_size"],
                instance_type=config["instance"]["instance_type"],
                namespace=config["instance"]["namespace"],
                image_url=config["instance"].get("image_url", None),
                env_vars=config["instance"].get("env_vars", None),
            )
        return InferenceModelConfig(model=config["base_params"]["endpoint_name"])

    if config["type"] == "base":
        # Creating the multichoice space parameters
        # We need to take into account possible conversion issues from our different input formats
        multichoice_continuations_start_space = boolstring_to_bool(
            config["generation"]["multichoice_continuations_start_space"]
        )

        if multichoice_continuations_start_space is not None:
            if multichoice_continuations_start_space:
                hlog(
                    "You set `multichoice_continuations_start_space` to true. This will force multichoice continuations to use a starting space"
                )
            else:
                hlog(
                    "You set `multichoice_continuations_start_space` to false. This will remove a leading space from multichoice continuations, if present."
                )

        # Creating optional quantization configuration
        if config["base_params"]["dtype"] == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif config["base_params"]["dtype"] == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        # We extract the model args
        args_dict = {k.split("=")[0]: k.split("=")[1] for k in config["base_params"]["model_args"].split(",")}

        # We store the relevant other args
        args_dict["base_model"] = config["merged_weights"]["base_model"]
        args_dict["compile"] = bool(config["base_params"]["compile"])
        args_dict["dtype"] = config["base_params"]["dtype"]
        args_dict["accelerator"] = accelerator
        args_dict["quantization_config"] = quantization_config
        args_dict["batch_size"] = override_batch_size
        args_dict["multichoice_continuations_start_space"] = multichoice_continuations_start_space
        args_dict["use_chat_template"] = use_chat_template

        # Keeping only non null params
        args_dict = {k: v for k, v in args_dict.items() if v is not None}

        if config["merged_weights"]["delta_weights"]:
            if config["merged_weights"]["base_model"] is None:
                raise ValueError("You need to specify a base model when using delta weights")
            return DeltaModelConfig(**args_dict)
        if config["merged_weights"]["adapter_weights"]:
            if config["merged_weights"]["base_model"] is None:
                raise ValueError("You need to specify a base model when using adapter weights")
            return AdapterModelConfig(**args_dict)
        if config["merged_weights"]["base_model"] not in ["", None]:
            raise ValueError("You can't specify a base model if you are not using delta/adapter weights")
        return BaseModelConfig(**args_dict)

    raise ValueError(f"Unknown model type in your model config file: {config['type']}")

from argparse import Namespace
from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import AutoConfig, BitsAndBytesConfig, GPTQConfig, PretrainedConfig

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.utils import _get_model_sha
from lighteval.utils import (
    NO_AUTOGPTQ_ERROR_MSG,
    NO_BNB_ERROR_MSG,
    NO_PEFT_ERROR_MSG,
    is_accelerate_available,
    is_autogptq_available,
    is_bnb_available,
    is_peft_available,
)


if is_accelerate_available():
    from accelerate import Accelerator


@dataclass
class EnvConfig:
    """
    Configuration class for environment settings.

    Attributes:
        cache_dir (str): directory for caching data.
        token (str): authentication token used for accessing the HuggingFace Hub.
    """

    cache_dir: str = None
    token: str = None

    """Args:
    pretrained (str):
        HuggingFace Hub model ID name or the path to a pre-trained
        model to load. This is effectively the `pretrained_model_name_or_path`
        argument of `from_pretrained` in the HuggingFace `transformers` API.
    add_special_tokens (bool, optional, defaults to True):
        Whether to add special tokens to the input sequences. If `None`, the
        default value will be set to `True` for seq2seq models (e.g. T5) and
        `False` for causal models.
    > Large model loading `accelerate` arguments
    model_parallel (bool, optional, defaults to False):
        True/False: force to uses or not the `accelerate` library to load a large
        model across multiple devices.
        Default: None which correspond to comparing the number of process with
            the number of GPUs. If it's smaller => model-parallelism, else not.
    dtype (Union[str, torch.dtype], optional, defaults to None):):
        Converts the model weights to `dtype`, if specified. Strings get
        converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
        Use `dtype="auto"` to derive the type from the model's weights.
    """


@dataclass
class BaseModelConfig:
    """
    Base configuration class for models.

    Attributes:
        pretrained (str): HuggingFace Hub model ID name or the path to a
            pre-trained model to load. This is effectively the
            `pretrained_model_name_or_path` argument of `from_pretrained` in the
            HuggingFace `transformers` API.
        accelerator (Accelerator): accelerator to use for model training.
        tokenizer (Optional[str]): HuggingFace Hub tokenizer ID that will be
            used for tokenization.
        multichoice_continuations_start_space (Optional[bool]): Whether to add a
            space at the start of each continuation in multichoice generation.
            For example, context: "What is the capital of France?" and choices: "Paris", "London".
            Will be tokenized as: "What is the capital of France? Paris" and "What is the capital of France? London".
        subfolder (Optional[str]): The subfolder within the model repository.
        revision (str): The revision of the model.
        batch_size (int): The batch size for model training.
        max_gen_toks (Optional[int]): The maximum number of tokens to generate.
        max_length (Optional[int]): The maximum length of the generated output.
        add_special_tokens (bool, optional, defaults to True): Whether to add special tokens to the input sequences.
           If `None`, the default value will be set to `True` for seq2seq models (e.g. T5) and
            `False` for causal models.
        model_parallel (Optional[bool]): Whether to use model parallelism.
        dtype (Optional[Union[str, torch.dtype]]): data type of the model.
        device (Union[int, str]): device to use for model training.
        quantization_config (Optional[BitsAndBytesConfig]): quantization
            configuration for the model. Needed for 4-bit and 8-bit precision.
        load_in_8bit (bool): Whether to load the model in 8-bit precision.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.
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
    load_in_8bit: bool = None
    load_in_4bit: bool = None
    trust_remote_code: bool = False

    def __post_init__(self):
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
        if getattr(auto_config, "quantization_config", False) and self.quantization_config is None:
            if not is_autogptq_available():
                raise ImportError(NO_AUTOGPTQ_ERROR_MSG)
            hlog(
                "`quantization_config` is None but was found in the model's config, using the one found in config.json"
            )
            self.quantization_config = GPTQConfig(**auto_config.quantization_config, disable_exllama=True)

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
class TGIModelConfig:
    inference_server_address: str
    inference_server_auth: str


@dataclass
class InferenceModelConfig:
    model: str


@dataclass
class InferenceEndpointModelConfig:
    name: str
    repository: str
    accelerator: str
    vendor: str
    region: str
    instance_size: str
    instance_type: str
    framework: str = "pytorch"
    endpoint_type: str = "protected"
    should_reuse_existing: bool = False


def create_model_config(args: Namespace, accelerator: Union["Accelerator", None]) -> BaseModelConfig:  # noqa: C901
    """
    Create a model configuration based on the provided arguments.

    Args:
        args (Namespace): command-line arguments.
        accelerator (Union[Accelerator, None]): accelerator to use for model training.

    Returns:
        BaseModelConfig: model configuration.

    Raises:
        ValueError: If both an inference server address and model arguments are provided.
     ValueError: If multichoice continuations both should start with a space and should not start with a space.
        ValueError: If a base model is not specified when using delta weights or adapter weights.
        ValueError: If a base model is specified when not using delta weights or adapter weights.
    """
    if args.inference_server_address is not None and args.model_args is not None:
        raise ValueError("You cannot both use an inference server and load a model from its checkpoint.")
    if args.inference_server_address is not None and args.endpoint_model_name is not None:
        raise ValueError("You cannot both use a local inference server and load a model from an inference endpoint.")
    if args.endpoint_model_name is not None and args.model_args is not None:
        raise ValueError("You cannot both load a model from its checkpoint and from an inference endpoint.")

    # TGI
    if args.inference_server_address is not None:
        return TGIModelConfig(
            inference_server_address=args.inference_server_address, inference_server_auth=args.inference_server_auth
        )

    # Endpoint
    if args.endpoint_model_name:
        if args.reuse_existing or args.vendor is not None:
            model = args.endpoint_model_name.split("/")[1].lower()
            return InferenceEndpointModelConfig(
                name=f"{model}-lighteval",
                repository=args.endpoint_model_name,
                accelerator=args.accelerator,
                region=args.region,
                vendor=args.vendor,
                instance_size=args.instance_size,
                instance_type=args.instance_type,
                should_reuse_existing=args.reuse_existing,
            )
        return InferenceModelConfig(model=args.endpoint_model_name)

    # Base
    multichoice_continuations_start_space = args.multichoice_continuations_start_space
    if not multichoice_continuations_start_space and not args.no_multichoice_continuations_start_space:
        multichoice_continuations_start_space = None
    if args.multichoice_continuations_start_space and args.no_multichoice_continuations_start_space:
        raise ValueError(
            "You cannot force both the multichoice continuations to start with a space and not to start with a space"
        )

    if "load_in_4bit=True" in args.model_args:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif "load_in_8bit=True" in args.model_args:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    # We extract the model args
    args_dict = {k.split("=")[0]: k.split("=")[1] for k in args.model_args.split(",")}
    # We store the relevant other args
    args_dict["base_model"] = args.base_model
    args_dict["batch_size"] = args.override_batch_size
    args_dict["accelerator"] = accelerator
    args_dict["quantization_config"] = quantization_config
    args_dict["dtype"] = args.model_dtype
    args_dict["multichoice_continuations_start_space"] = multichoice_continuations_start_space

    # Keeping only non null params
    args_dict = {k: v for k, v in args_dict.items() if v is not None}

    if args.delta_weights:
        if args.base_model is None:
            raise ValueError("You need to specify a base model when using delta weights")
        return DeltaModelConfig(**args_dict)
    if args.adapter_weights:
        if args.base_model is None:
            raise ValueError("You need to specify a base model when using adapter weights")
        return AdapterModelConfig(**args_dict)
    if args.base_model is not None:
        raise ValueError("You can't specifify a base model if you are not using delta/adapter weights")
    return BaseModelConfig(**args_dict)

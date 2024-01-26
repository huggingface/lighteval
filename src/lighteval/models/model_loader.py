from dataclasses import dataclass
from typing import Optional, Tuple, Union

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.adapter_model import AdapterModel
from lighteval.models.base_model import BaseModel
from lighteval.models.delta_model import DeltaModel
from lighteval.models.inference_client import ModelClient
from lighteval.models.model_config import (
    AdapterModelConfig,
    BaseModelConfig,
    DeltaModelConfig,
    EnvConfig,
    TGIModelConfig,
)
from lighteval.utils import NO_TGI_ERROR_MSG, is_accelerate_available, is_tgi_available


if is_accelerate_available():
    from accelerate.utils import calculate_maximum_sizes, convert_bytes


@dataclass
class ModelInfo:
    model_name: str
    model_sha: Optional[str] = None
    model_dtype: Optional[str] = None
    model_size: Optional[str] = None


def load_model(  # noqa: C901
    config: Union[BaseModelConfig, AdapterModelConfig, DeltaModelConfig, TGIModelConfig], env_config: EnvConfig
) -> Tuple[Union[BaseModel, AdapterModel, DeltaModel, ModelClient], ModelInfo]:
    """Will load either a model from an inference server or a model from a checkpoint. depending
    on the arguments passed to the program.

    Args:
        args (Namespace): arguments passed to the program
        accelerator (Accelerator): Accelerator that will be used by the model

    Raises:
        ValueError: If you try to load a model from an inference server and from a checkpoint at the same time
        ValueError: If you try to have both the multichoice continuations start with a space and not to start with a space
        ValueError: If you did not specify a base model when using delta weights or adapter weights

    Returns:
        Union[BaseModel, AdapterModel, DeltaModel, ModelClient]: The model that will be evaluated
    """
    ## Inference server loading
    if isinstance(config, TGIModelConfig):
        return load_model_with_tgi(config)

    if isinstance(config, BaseModelConfig):
        return load_model_with_accelerate_or_default(config=config, env_config=env_config)


def load_model_with_tgi(config: TGIModelConfig):
    if not is_tgi_available():
        raise ImportError(NO_TGI_ERROR_MSG)

    hlog(f"Load model from inference server: {config.inference_server_address}")
    model = ModelClient(address=config.inference_server_address, auth_token=config.inference_server_auth)
    model_name = str(model.model_info["model_id"])
    model_sha = model.model_info["model_sha"]
    model_precision = model.model_info["dtype"]
    model_size = -1
    model_info = ModelInfo(
        model_name=model_name,
        model_sha=model_sha,
        model_dtype=model_precision,
        model_size=model_size,
    )
    return model, model_info


def load_model_with_accelerate_or_default(
    config: Union[AdapterModelConfig, BaseModelConfig, DeltaModelConfig], env_config: EnvConfig
):
    if isinstance(config, AdapterModelConfig):
        model = AdapterModel(config=config, env_config=env_config)
    elif isinstance(config, DeltaModelConfig):
        model = DeltaModel(config=config, env_config=env_config)
    else:
        model = BaseModel(config=config, env_config=env_config)

    model_name = model.model_name
    model_sha = model.model_sha
    model_precision = str(model.precision)
    if is_accelerate_available():
        model_size, _ = calculate_maximum_sizes(model.model)
        model_size = convert_bytes(model_size)
    else:
        model_size = -1
    model_info = ModelInfo(
        model_name=model_name,
        model_sha=model_sha,
        model_dtype=model_precision,
        model_size=model_size,
    )
    hlog(f"Model info: {model_info}")

    return model, model_info

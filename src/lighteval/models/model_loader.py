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

from typing import Union

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.adapter_model import AdapterModel
from lighteval.models.base_model import BaseModel
from lighteval.models.delta_model import DeltaModel
from lighteval.models.dummy_model import DummyModel
from lighteval.models.endpoint_model import InferenceEndpointModel
from lighteval.models.model_config import (
    AdapterModelConfig,
    BaseModelConfig,
    DeltaModelConfig,
    DummyModelConfig,
    InferenceEndpointModelConfig,
    InferenceModelConfig,
    TGIModelConfig,
    VLLMModelConfig,
)
from lighteval.models.tgi_model import ModelClient
from lighteval.models.vllm_model import VLLMModel
from lighteval.utils.imports import NO_TGI_ERROR_MSG, NO_VLLM_ERROR_MSG, is_tgi_available, is_vllm_available
from lighteval.utils.utils import EnvConfig


def load_model(  # noqa: C901
    config: Union[
        BaseModelConfig,
        AdapterModelConfig,
        DeltaModelConfig,
        TGIModelConfig,
        InferenceEndpointModelConfig,
        DummyModelConfig,
        VLLMModelConfig,
    ],
    env_config: EnvConfig,
) -> Union[BaseModel, AdapterModel, DeltaModel, ModelClient, DummyModel]:
    """Will load either a model from an inference server or a model from a checkpoint, depending
    on the config type.

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
    # Inference server loading
    if isinstance(config, TGIModelConfig):
        return load_model_with_tgi(config)

    if isinstance(config, InferenceEndpointModelConfig) or isinstance(config, InferenceModelConfig):
        return load_model_with_inference_endpoints(config, env_config=env_config)

    if isinstance(config, BaseModelConfig):
        return load_model_with_accelerate_or_default(config=config, env_config=env_config)

    if isinstance(config, DummyModelConfig):
        return load_dummy_model(config=config, env_config=env_config)

    if isinstance(config, VLLMModelConfig):
        return load_model_with_accelerate_or_default(config=config, env_config=env_config)


def load_model_with_tgi(config: TGIModelConfig):
    if not is_tgi_available():
        raise ImportError(NO_TGI_ERROR_MSG)

    hlog(f"Load model from inference server: {config.inference_server_address}")
    model = ModelClient(
        address=config.inference_server_address, auth_token=config.inference_server_auth, model_id=config.model_id
    )
    return model


def load_model_with_inference_endpoints(config: InferenceEndpointModelConfig, env_config: EnvConfig):
    hlog("Spin up model using inference endpoint.")
    model = InferenceEndpointModel(config=config, env_config=env_config)
    return model


def load_model_with_accelerate_or_default(
    config: Union[AdapterModelConfig, BaseModelConfig, DeltaModelConfig], env_config: EnvConfig
):
    if isinstance(config, AdapterModelConfig):
        model = AdapterModel(config=config, env_config=env_config)
    elif isinstance(config, DeltaModelConfig):
        model = DeltaModel(config=config, env_config=env_config)
    elif isinstance(config, VLLMModelConfig):
        if not is_vllm_available():
            raise ImportError(NO_VLLM_ERROR_MSG)
        model = VLLMModel(config=config, env_config=env_config)
        return model
    else:
        model = BaseModel(config=config, env_config=env_config)

    return model


def load_dummy_model(config: DummyModelConfig, env_config: EnvConfig):
    return DummyModel(config=config, env_config=env_config)

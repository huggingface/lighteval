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
from typing import Union

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
from lighteval.models.endpoints.endpoint_model import (
    InferenceEndpointModel,
    InferenceEndpointModelConfig,
    ServerlessEndpointModelConfig,
)
from lighteval.models.endpoints.inference_providers_model import (
    InferenceProvidersClient,
    InferenceProvidersModelConfig,
)
from lighteval.models.endpoints.openai_model import OpenAIClient, OpenAIModelConfig
from lighteval.models.endpoints.tgi_model import ModelClient, TGIModelConfig
from lighteval.models.litellm_model import LiteLLMClient, LiteLLMModelConfig
from lighteval.models.sglang.sglang_model import SGLangModel, SGLangModelConfig
from lighteval.models.transformers.adapter_model import AdapterModel, AdapterModelConfig
from lighteval.models.transformers.delta_model import DeltaModel, DeltaModelConfig
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.models.transformers.vlm_transformers_model import VLMTransformersModel, VLMTransformersModelConfig
from lighteval.models.utils import ModelConfig
from lighteval.models.vllm.vllm_model import AsyncVLLMModel, VLLMModel, VLLMModelConfig
from lighteval.utils.imports import (
    NO_LITELLM_ERROR_MSG,
    NO_SGLANG_ERROR_MSG,
    NO_TGI_ERROR_MSG,
    NO_VLLM_ERROR_MSG,
    is_litellm_available,
    is_openai_available,
    is_sglang_available,
    is_tgi_available,
    is_vllm_available,
)


logger = logging.getLogger(__name__)


def load_model(  # noqa: C901
    config: ModelConfig,
) -> LightevalModel:
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
        Union[TransformersModel, AdapterModel, DeltaModel, ModelClient]: The model that will be evaluated
    """
    # Inference server loading
    if isinstance(config, TGIModelConfig):
        return load_model_with_tgi(config)

    if isinstance(config, InferenceEndpointModelConfig) or isinstance(config, ServerlessEndpointModelConfig):
        return load_model_with_inference_endpoints(config)

    if isinstance(config, TransformersModelConfig):
        return load_model_with_accelerate_or_default(config)

    if isinstance(config, VLMTransformersModelConfig):
        return load_model_with_accelerate_or_default(config)

    if isinstance(config, DummyModelConfig):
        return load_dummy_model(config)

    if isinstance(config, VLLMModelConfig):
        return load_model_with_accelerate_or_default(config)

    if isinstance(config, CustomModelConfig):
        return load_custom_model(config=config)

    if isinstance(config, SGLangModelConfig):
        return load_sglang_model(config)

    if isinstance(config, OpenAIModelConfig):
        return load_openai_model(config)

    if isinstance(config, LiteLLMModelConfig):
        return load_litellm_model(config)

    if isinstance(config, InferenceProvidersModelConfig):
        return load_inference_providers_model(config=config)


def load_model_with_tgi(config: TGIModelConfig):
    if not is_tgi_available():
        raise ImportError(NO_TGI_ERROR_MSG)

    logger.info(f"Load model from inference server: {config.inference_server_address}")
    model = ModelClient(
        address=config.inference_server_address, auth_token=config.inference_server_auth, model_id=config.model_id
    )
    return model


def load_litellm_model(config: LiteLLMModelConfig):
    if not is_litellm_available():
        raise ImportError(NO_LITELLM_ERROR_MSG)

    model = LiteLLMClient(config)
    return model


def load_openai_model(config: OpenAIModelConfig):
    if not is_openai_available():
        raise ImportError()

    model = OpenAIClient(config)

    return model


def load_custom_model(config: CustomModelConfig):
    logger.warning(f"Executing custom model code loaded from {config.model_definition_file_path}.")

    import importlib.util

    # Load the Python file
    spec = importlib.util.spec_from_file_location("custom_model_module", config.model_definition_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load file: {config.model_definition_file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first class that inherits from LightevalModel
    model_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, LightevalModel) and attr != LightevalModel:
            model_class = attr
            break

    if model_class is None:
        raise ValueError(f"No class inheriting from LightevalModel found in {config.model_definition_file_path}")

    model = model_class(config)

    return model


def load_model_with_inference_endpoints(config: Union[InferenceEndpointModelConfig, ServerlessEndpointModelConfig]):
    logger.info("Spin up model using inference endpoint.")
    model = InferenceEndpointModel(config=config)
    return model


def load_model_with_accelerate_or_default(
    config: Union[
        AdapterModelConfig, TransformersModelConfig, DeltaModelConfig, VLLMModelConfig, VLMTransformersModelConfig
    ],
):
    if isinstance(config, AdapterModelConfig):
        model = AdapterModel(config=config)
    elif isinstance(config, DeltaModelConfig):
        model = DeltaModel(config=config)
    elif isinstance(config, VLLMModelConfig):
        if not is_vllm_available():
            raise ImportError(NO_VLLM_ERROR_MSG)
        if config.is_async:
            model = AsyncVLLMModel(config=config)
        else:
            model = VLLMModel(config=config)
    elif isinstance(config, VLMTransformersModelConfig):
        model = VLMTransformersModel(config=config)
    else:
        model = TransformersModel(config=config)

    return model


def load_dummy_model(config: DummyModelConfig):
    return DummyModel(config=config)


def load_inference_providers_model(config: InferenceProvidersModelConfig):
    return InferenceProvidersClient(config=config)


def load_sglang_model(config: SGLangModelConfig):
    if not is_sglang_available():
        raise ImportError(NO_SGLANG_ERROR_MSG)

    return SGLangModel(config=config)

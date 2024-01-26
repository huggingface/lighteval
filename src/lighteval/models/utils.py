import os
from typing import TYPE_CHECKING, Optional, Union

import torch
from huggingface_hub import HfApi
from transformers import AutoConfig


if TYPE_CHECKING:
    from lighteval.models.model_config import BaseModelConfig


def _get_dtype(dtype: Union[str, torch.dtype], config: Optional[AutoConfig] = None) -> torch.dtype:
    """
    Get the torch dtype based on the input arguments.

    Args:
        dtype (Union[str, torch.dtype]): The desired dtype. Can be a string or a torch dtype.
        config (Optional[transformers.AutoConfig]): The model config object. Defaults to None.

    Returns:
        torch.dtype: The torch dtype based on the input arguments.
    """

    if dtype is None and config is not None:
        # For quantized models
        if hasattr(config, "quantization_config"):
            _torch_dtype = None  # must be inferred
        else:
            _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


def _simplify_name(name_or_path: str) -> str:
    """
    If the model is loaded from disk, then the name will have the following format:
    /p/a/t/h/models--org--model_name/revision/model_files
    This function return the model_name as if it was loaded from the hub:
    org/model_name

    Args:
        name_or_path (str): The name or path to be simplified.

    Returns:
        str: The simplified name.
    """
    if os.path.isdir(name_or_path) or os.path.isfile(name_or_path):  # Loading from disk
        simple_name_list = name_or_path.split("/")
        # The following manages files stored on disk, loaded with the hub model format:
        # /p/a/t/h/models--org--model_name/revision/model_files
        if any("models--" in item for item in simple_name_list):  # Hub format
            simple_name = [item for item in simple_name_list if "models--" in item][0]
            simple_name = simple_name.replace("models--", "").replace("--", "/")
            return simple_name
        # This is for custom folders
        else:  # Just a folder, we don't know the shape
            return name_or_path.replace("/", "_")

    return name_or_path


def _get_precision(config: "BaseModelConfig", model_auto_config: AutoConfig):
    """
    Returns the precision of the model.

    Args:
        dtype (Union[str, torch.dtype]): The data type of the model.
        load_in_8bit (bool): Whether to load the weights in 8-bit precision.
        load_in_4bit (bool): Whether to load the weights in 4-bit precision.

    Returns:
        str: The selected precision for loading the model weights.
    """
    if config.load_in_8bit:
        return "8bit"
    elif config.load_in_4bit:
        return "4bit"
    else:
        return _get_dtype(config.dtype, model_auto_config)


def _get_model_sha(repo_id: str, revision: str):
    api = HfApi()
    try:
        model_info = api.model_info(repo_id=repo_id, revision=revision)
        return model_info.sha
    except Exception:
        return ""

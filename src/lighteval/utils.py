# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from typing import Any, Union

import numpy as np


def sanitize_numpy(example_dict: dict) -> dict:
    """
    Sanitizes a dictionary by converting any numpy generic types to their corresponding Python types.

    Args:
        example_dict (dict): The dictionary to be sanitized.

    Returns:
        dict: The sanitized dictionary with numpy generic types converted to Python types.
    """
    output_dict = {}
    for k, v in example_dict.items():
        if isinstance(v, np.generic):
            output_dict[k] = v.item()
        else:
            output_dict[k] = v
    return output_dict


def as_list(item: Union[list, tuple, Any]) -> list:
    """
    Convert the given item into a list.

    If the item is already a list, it is returned as is.
    If the item is a tuple, it is converted into a list.
    Otherwise, the item is wrapped in a list.

    Args:
        item (Union[list, tuple, Any]): The item to be converted.

    Returns:
        list: The converted list.

    """
    if isinstance(item, list):
        return item
    elif isinstance(item, tuple):
        return list(item)
    return [item]


def flatten(item: list[Union[list, str]]) -> list[str]:
    """
    Flattens a nested list of strings into a single flat list.

    Args:
        item (list[Union[list, str]]): The nested list to be flattened.

    Returns:
        list[str]: The flattened list of strings.
    """
    flat_item = []
    for sub_item in item:
        flat_item.extend(sub_item) if isinstance(sub_item, list) else flat_item.append(sub_item)
    return flat_item


def is_accelerate_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


NO_ACCELERATE_ERROR_MSG = "You requested the use of accelerate for this evaluation, but it is not available in your current environement. Please install it using pip."


def is_tgi_available() -> bool:
    return importlib.util.find_spec("text-generation") is not None


NO_TGI_ERROR_MSG = "You are trying to start a text generation inference endpoint, but text-generation is not present in your local environement. Please install it using pip."


def is_nanotron_available() -> bool:
    return importlib.util.find_spec("nanotron") is not None


NO_NANOTRON_ERROR_MSG = "YYou requested the use of nanotron for this evaluation, but it is not available in your current environement. Please install it using pip."


def is_optimum_available() -> bool:
    return importlib.util.find_spec("optimum") is not None


def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


NO_BNB_ERROR_MSG = "You are trying to load a model quantized with `bitsandbytes`, which is not available in your local environement. Please install it using pip."


def is_autogptq_available() -> bool:
    return importlib.util.find_spec("auto-gptq") is not None


NO_AUTOGPTQ_ERROR_MSG = "You are trying to load a model quantized with `auto-gptq`, which is not available in your local environement. Please install it using pip."


def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None


NO_PEFT_ERROR_MSG = "You are trying to use adapter weights models, for which you need `peft`, which is not available in your environment. Please install it using pip."

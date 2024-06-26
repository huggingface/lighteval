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
from dataclasses import asdict, is_dataclass
from typing import Any, Union

import numpy as np


def flatten_dict(nested: dict, sep="/") -> dict:
    """Flatten dictionary, list, tuple and concatenate nested keys with separator."""

    def clean_markdown(v: str) -> str:
        return v.replace("|", "_").replace("\n", "_") if isinstance(v, str) else v  # Need this for markdown

    def rec(nest: dict, prefix: str, into: dict):
        for k, v in sorted(nest.items()):
            # if sep in k:
            #     raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):
                rec(v, prefix + k + sep, into)
            elif isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    if isinstance(vv, dict):
                        rec(vv, prefix + k + sep + str(i) + sep, into)
                    else:
                        vv = (
                            vv.replace("|", "_").replace("\n", "_") if isinstance(vv, str) else vv
                        )  # Need this for markdown
                        into[prefix + k + sep + str(i)] = vv.tolist() if isinstance(vv, np.ndarray) else vv
            elif isinstance(v, np.ndarray):
                into[prefix + k + sep + str(i)] = v.tolist()
            else:
                v = clean_markdown(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def clean_s3_links(value: str) -> str:
    """Cleans and formats s3 bucket links for better display in the result table (nanotron models)

    Args:
        value (str): path to clean

    Returns:
        str : cleaned path
    """
    s3_bucket, s3_prefix = str(value).replace("s3://", "").split("/", maxsplit=1)
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    link_str = f"https://s3.console.aws.amazon.com/s3/buckets/{s3_bucket}?prefix={s3_prefix}"
    value = f'<a href="{link_str}" target="_blank"> {value} </a>'
    return value


def obj_to_markdown(obj, convert_s3_links: bool = True) -> str:
    """Convert a (potentially nested) dataclass object or a dict in a readable markdown string for logging"""
    from pytablewriter import MarkdownTableWriter

    if is_dataclass(obj):
        obj = asdict(obj)
    config_dict = flatten_dict(obj)

    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Key", "Value"]

    values = []
    for key, value in config_dict.items():
        if convert_s3_links and "s3://" in str(value):
            value = clean_s3_links(value)
        values.append([key, value])
    md_writer.value_matrix = values

    return md_writer.dumps()


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


NO_NANOTRON_ERROR_MSG = "You requested the use of nanotron for this evaluation, but it is not available in your current environement. Please install it using pip."


def is_optimum_available() -> bool:
    return importlib.util.find_spec("optimum") is not None


def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


NO_BNB_ERROR_MSG = "You are trying to load a model quantized with `bitsandbytes`, which is not available in your local environement. Please install it using pip."


def is_autogptq_available() -> bool:
    return importlib.util.find_spec("auto_gptq") is not None


NO_AUTOGPTQ_ERROR_MSG = "You are trying to load a model quantized with `auto-gptq`, which is not available in your local environement. Please install it using pip."


def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None


NO_PEFT_ERROR_MSG = "You are trying to use adapter weights models, for which you need `peft`, which is not available in your environment. Please install it using pip."


def can_load_extended_tasks() -> bool:
    imports = []
    for package in ["langdetect"]:
        imports.append(importlib.util.find_spec(package))

    return all(cur_import is not None for cur_import in imports)


CANNOT_USE_EXTENDED_TASKS_MSG = "If you want to use extended_tasks, make sure you installed their dependencies using `pip install -e .[extended_tasks]`."

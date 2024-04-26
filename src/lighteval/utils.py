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
import importlib
import json
import os

import pkg_resources



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


def load_tasks_table_extended(module_name: any) -> list:
    """
    load the module module_name

    Args:
    - module_name the name of the module we want to load
    Returns:
    - TASKS_TABLE: a list of the task in the module
    """
    module_path = f"lighteval.tasks.extended.{module_name}.main"
    module_loaded = importlib.import_module(module_path)
    tasks_list = None
    try:
        tasks_list = module_loaded.TASKS_TABLE
    except Exception as e:
        print(e)
    return tasks_list if tasks_list is not None else []


def get_tasks_table_json() -> list:
    """
    Fetch tasks/tasks_table.jsonl
    Returns
    - a list of all the tasks in tasks/tasks_table.jsonl
    """
    tasks = []
    # Handling tasks_table.jsonl
    # Get the path to the resource file
    tasks_table_path = pkg_resources.resource_filename("lighteval", "tasks/tasks_table.jsonl")
    with open(tasks_table_path) as jsonl_tasks_table:
        jsonl_tasks_table_content = jsonl_tasks_table.read()
        for jline in jsonl_tasks_table_content.splitlines():
            tasks.append(json.loads(jline))
    return tasks


def get_extended_tasks() -> list:
    """
    Fetch all the tasks in the extended suite
    Returns
    - a list of all the extended tasks
    """
    tasks_extended = []
    extended_tasks_dir = pkg_resources.resource_filename("lighteval", "tasks/extended")
    for root, dirs, files in os.walk(extended_tasks_dir):
        for file in files:
            if file == "main.py":
                module_name = os.path.basename(root)
                tasks_table = load_tasks_table_extended(module_name)
                tasks_extended += tasks_table
    return tasks_extended


def group_by_suite(tasks: list, tasks_extended: list) -> dict:
    """
    Group tasks by suite and sort them alphabetically
    Args:
    - tasks: list of tasks in tasks/tasks_table.jsonl
    - tasks_extended: list of extended tasks
    Returns:
    - a dict of tasks grouped by suite
    """
    grouped_by_suite = {}
    for task in tasks:
        for suite in task["suite"]:
            if suite not in grouped_by_suite.keys():
                grouped_by_suite[suite] = [task["name"]]
            else:
                grouped_by_suite[suite].append(task["name"])
                grouped_by_suite[suite].sort()

    grouped_by_suite["extended"] = []
    # Adding extended suite
    for task in tasks_extended:
        grouped_by_suite["extended"].append(task["name"])
    grouped_by_suite["extended"].sort()
    return grouped_by_suite


def list_tasks_command():
    """
    List all the available tasks in tasks_table.jsonl and the extended directory
    Assumes the existence of TASKS_TABLE in the main.py file for each extended
    tasks in tasks/extended
    """
    try:
        # Handling tasks_table.jsonl
        tasks = get_tasks_table_json()

        # Handling extended tasks
        tasks_extended = get_extended_tasks()

        # Grouping by suite the tasks
        grouped_by_suite = group_by_suite(tasks, tasks_extended)

        # Print tasks
        print("Available tasks: (Grouped by suite)\n")
        for suite, task_list in grouped_by_suite.items():
            print("- " + suite)
            for task in task_list:
                print("\t - " + task)
    except Exception as e:
        print("Error: ", e)


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

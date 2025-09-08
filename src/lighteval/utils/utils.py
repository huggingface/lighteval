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
from dataclasses import asdict, is_dataclass
from typing import TypeVar, Union

import numpy as np
from pytablewriter import MarkdownTableWriter


def flatten_dict(nested: dict, sep: str = "/") -> dict:
    """Flattens a nested dictionary structure into a single-level dictionary.

    Recursively traverses nested dictionaries, lists, and tuples, concatenating keys with a separator
    to create unique keys for each value. Handles numpy arrays and sanitizes markdown-incompatible characters.

    Args:
        nested (dict): The nested dictionary structure to flatten.
        sep (str, optional): The separator to use between nested keys. Defaults to "/".

    Returns:
        dict: A flattened dictionary where all values are at the root level and keys indicate
              the original nesting structure using the separator.

    Examples:
        >>> flatten_dict({'a': {'b': 1, 'c': [{'d': 2}, {'e': 3}]}})
        {'a/b': 1, 'a/c/0/d': 2, 'a/c/1/e': 3}
    """

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
    """Converts S3 bucket links to clickable HTML links for better display in result tables.

    Takes an S3 URI and converts it to a clickable HTML link that opens the AWS S3 console
    at the specific bucket and prefix location. This is particularly useful for nanotron models
    and other S3-hosted model artifacts.

    Args:
        value (str): The S3 URI to convert (e.g., "s3://my-bucket/path/to/model").

    Returns:
        str: HTML anchor tag with the original S3 URI as display text and a link to the AWS S3 console.

    Examples:
        >>> clean_s3_links("s3://my-bucket/models/gpt2")
        '<a href="https://s3.console.aws.amazon.com/s3/buckets/my-bucket?prefix=models/gpt2/" target="_blank"> s3://my-bucket/models/gpt2 </a>'
    """
    s3_bucket, s3_prefix = str(value).replace("s3://", "").split("/", maxsplit=1)
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    link_str = f"https://s3.console.aws.amazon.com/s3/buckets/{s3_bucket}?prefix={s3_prefix}"
    value = f'<a href="{link_str}" target="_blank"> {value} </a>'
    return value


def obj_to_markdown(obj, convert_s3_links: bool = True) -> str:
    """Converts a dataclass object or dictionary to a readable markdown table for logging.

    Flattens the object structure and creates a markdown table with key-value pairs.
    Optionally converts S3 links to clickable HTML links for better display.

    Args:
        obj: A dataclass object or dictionary to convert to markdown.
        convert_s3_links (bool, optional): Whether to convert S3 URIs to clickable links.
                                          Defaults to True.

    Returns:
        str: A markdown-formatted table string with the object's key-value pairs.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     model_name: str = "gpt2"
        ...     batch_size: int = 32
        >>> obj_to_markdown(Config())
        '| Key | Value |\n|-----|-------|\n| model_name | gpt2 |\n| batch_size | 32 |'
    """
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
    """Sanitizes a dictionary by converting numpy generic types to their corresponding Python types.

    Numpy generic types (like np.int64, np.float32, etc.) can cause issues with JSON serialization
    and other operations. This function converts them to standard Python types.

    Args:
        example_dict (dict): The dictionary to be sanitized.

    Returns:
        dict: The sanitized dictionary with numpy generic types converted to Python types.

    Examples:
        >>> import numpy as np
        >>> data = {'score': np.float32(0.95), 'count': np.int64(42)}
        >>> sanitize_numpy(data)
        {'score': 0.95, 'count': 42}
    """
    output_dict = {}
    for k, v in example_dict.items():
        if isinstance(v, np.generic):
            output_dict[k] = v.item()
        else:
            output_dict[k] = v
    return output_dict


ListLikeTypeVar = TypeVar("ListLikeTypeVar")
ListLike = list[ListLikeTypeVar] | tuple[ListLikeTypeVar, ...]


ElementType = TypeVar("ElementType")


def as_list(item: ListLike[ElementType] | ElementType) -> list[ElementType]:
    """Converts the given item into a list.

    If the item is already a list, it is returned as is. If the item is a tuple,
    it is converted into a list. Otherwise, the item is wrapped in a list.

    Args:
        item: The item to be converted. Can be a list, tuple, or any other type.

    Returns:
        list: The converted list.

    Examples:
        >>> as_list([1, 2, 3])
        [1, 2, 3]
        >>> as_list((1, 2, 3))
        [1, 2, 3]
        >>> as_list("single_item")
        ['single_item']
        >>> as_list(42)
        [42]
    """
    if isinstance(item, list):
        return item

    elif isinstance(item, tuple):
        return list(item)

    return [item]


def flatten(item: list[Union[list, str]]) -> list[str]:
    """Flattens a nested list of strings into a single flat list.

    Recursively flattens nested lists, extracting all string elements into a single-level list.
    Non-list elements are treated as strings and added directly.

    Args:
        item: The nested list to be flattened. Can contain strings and nested lists.

    Returns:
        list[str]: The flattened list of strings.

    Examples:
        >>> flatten([["a", "b"], "c", ["d", ["e", "f"]]])
        ['a', 'b', 'c', 'd', 'e', 'f']
        >>> flatten(["simple", "list"])
        ['simple', 'list']
    """
    flat_item = []
    for sub_item in item:
        flat_item.extend(sub_item) if isinstance(sub_item, list) else flat_item.append(sub_item)
    return flat_item


def make_results_table(result_dict):
    """Generates a markdown table from evaluation results.

    Creates a formatted markdown table displaying task results with metrics, values,
    and standard errors. The table includes columns for task name, version, metric,
    value, and standard error.

    Args:
        result_dict (dict): Dictionary containing evaluation results with the structure:
            - 'results': Dict mapping task names to metric dictionaries
            - 'versions': Dict mapping task names to version strings

    Returns:
        str: A markdown-formatted table string displaying the results.

    Examples:
        >>> results = {
        ...     'results': {
        ...         'squad': {'accuracy': 0.85, 'accuracy_stderr': 0.02},
        ...         'glue': {'f1': 0.92, 'f1_stderr': 0.01}
        ...     },
        ...     'versions': {'squad': 'v2.0', 'glue': 'v1.0'}
        ... }
        >>> table = make_results_table(results)
        # Returns markdown table with task, version, metric, value, ±, stderr columns
    """
    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k in sorted(result_dict["results"].keys()):
        dic = result_dict["results"][k]
        version = result_dict["versions"][k] if k in result_dict["versions"] else ""
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values

    return md_writer.dumps()


def safe_divide(numerator: np.ndarray, denominator: float, default_value: float = 0.0) -> np.ndarray:
    return np.where(denominator != 0, numerator / denominator, default_value)


def remove_reasoning_tags(text: str, tag_pairs: list[tuple[str, str]]) -> str:
    """Removes all instances of reasoning tag pairs from text.

    Iteratively removes content between specified start and end tag pairs.
    This is useful for cleaning model outputs that contain reasoning sections
    that should be excluded from evaluation.

    See: https://github.com/huggingface/lighteval/issues/790

    Args:
        text (str): The input text containing reasoning tags to remove.
        tag_pairs (list[tuple[str, str]]): List of (start_tag, end_tag) pairs to remove.

    Returns:
        str: The text with all reasoning tag content removed.

    Examples:
        >>> text = "<think> Reasoning section </think> Answer section"
        >>> tag_pairs = [("<think>", "</think>")]
        >>> remove_reasoning_tags(text, tag_pairs)
        ' Answer section'

        >>> text = "<reasoning>Step 1</reasoning>Answer<reasoning>Step 2</reasoning>"
        >>> tag_pairs = [("<reasoning>", "</reasoning>")]
        >>> remove_reasoning_tags(text, tag_pairs)
        'Answer'
    """
    result = text

    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag) :]
            else:
                break

    return result

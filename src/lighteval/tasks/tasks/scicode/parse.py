"""Parsing utilities for SciCode.

Based on original implementation:
https://github.com/scicode-bench/SciCode
"""

import ast
import re
from pathlib import Path

import h5py
import scipy.sparse


def extract_function_name(function_header: str) -> str:
    """Extract function or class name from function header."""
    pattern = r"\bdef\s+(\w+)\s*\("
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)

    pattern = r"\bclass\s+(\w+)\s*[\(:]"
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)

    raise ValueError(f"Function name or class name not found in: {function_header}")


def get_function_from_code(code_string: str, function_name: str) -> str:
    """Extract specific function/class from code using AST."""
    if code_string is None:
        return ""
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception:
        return code_string
    return None


def _process_hdf5_sparse_matrix(group: h5py.Group):
    """Process an h5py Group containing sparse matrix data."""
    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        row = group["row"][()]
        col = group["col"][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif "blocksize" in group:
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        blocksize = tuple(group["blocksize"][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def _process_hdf5_list(group: h5py.Group) -> list:
    """Process an h5py Group containing list data."""
    result_list = []
    for key in group.keys():
        result_list.append(group[key][()])
    return result_list


def _process_hdf5_dict(group: h5py.Group) -> dict:
    """Process an h5py Group into a dictionary."""
    result_dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            if "sparse_matrix" in obj:
                result_dict[key] = _process_hdf5_sparse_matrix(obj["sparse_matrix"])
            else:
                result_dict[key] = _process_hdf5_datagroup(obj)
        elif isinstance(obj, h5py.Dataset):
            if isinstance(obj[()], bytes):
                result_dict[key] = obj[()].decode("utf-8", errors="strict")
            else:
                try:
                    tmp = float(key)
                    result_dict[tmp] = obj[()]
                except ValueError:
                    result_dict[key] = obj[()]
    return result_dict


def _process_hdf5_datagroup(group: h5py.Group):
    """Process an h5py Group, handling special cases (list, sparse_matrix) or dict."""
    if "list" in group:
        return _process_hdf5_list(group["list"])
    elif "sparse_matrix" in group:
        return _process_hdf5_sparse_matrix(group["sparse_matrix"])
    else:
        return _process_hdf5_dict(group)


def extract_targets(step_id: str, num_tests: int, h5py_file: str | Path) -> tuple:
    """Extract target values from h5py file for a given step."""
    if isinstance(step_id, tuple):
        step_id = ".".join(str(x) for x in step_id)
    elif not isinstance(step_id, str):
        step_id = str(step_id)

    with h5py.File(h5py_file, "r") as f:
        if step_id not in f:
            raise ValueError(f"Step {step_id} not found in h5py file")
        targets = []
        for i in range(1, num_tests + 1):
            group_path = f"{step_id}/test{i}"

            try:
                if group_path not in f:
                    continue

                group = f[group_path]

                if "var1" in group:
                    var1 = group["var1"]
                    if isinstance(var1, h5py.Dataset):
                        target = var1[()]
                        targets.append(target)
                    elif isinstance(var1, h5py.Group):
                        target = _process_hdf5_datagroup(var1)
                        targets.append(target)
            except Exception:
                raise

    return tuple(targets)

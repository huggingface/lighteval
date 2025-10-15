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

"""
Automatically imports all task configs from the tasks/ directory.
This module dynamically loads all Python files in tasks/ and exposes their LightevalTaskConfig objects.
"""

import importlib
import time
from pathlib import Path


# Get the tasks directory
TASKS_DIR = Path(__file__).parent / "tasks"
TASKS_DIR_MULTILINGUAL = Path(__file__).parent / "multilingual" / "tasks"


def _extract_configs(module):
    configs = {}
    if hasattr(module, "TASKS_TABLE"):
        for config in getattr(module, "TASKS_TABLE"):
            configs[config.name] = config
    return configs


def _load_from_files(files, module_prefix: str):
    configs = {}
    for task_file in files:
        module_name = task_file.stem
        module = importlib.import_module(f"{module_prefix}.{module_name}")
        configs.update(_extract_configs(module))
    return configs


def _load_from_subdirs(subdirs):
    configs = {}
    for task_dir in subdirs:
        module_name = task_dir.name
        module = importlib.import_module(f"lighteval.tasks.tasks.{module_name}.main")
        configs.update(_extract_configs(module))
    return configs


def _load_all_task_configs():
    """Load all LightevalTaskConfig objects from all Python files in the tasks/ directory."""
    start_time = time.perf_counter()
    loaded_configs = {}

    # Get all Python files in the tasks directory (excluding __init__.py)
    task_files = [f for f in TASKS_DIR.glob("*.py") if f.name != "__init__.py"]
    # task_files_multilingual = [f for f in TASKS_DIR_MULTILINGUAL.glob("*.py") if f.name != "__init__.py"]

    # Also get all subdirectories with main.py files
    task_subdirs = [d for d in TASKS_DIR.iterdir() if d.is_dir() and (d / "main.py").exists()]

    loaded_configs.update(_load_from_files(task_files, "lighteval.tasks.tasks"))
    # loaded_configs.update(
    #     _load_from_files(task_files_multilingual, "lighteval.tasks.multilingual.tasks")
    # )
    loaded_configs.update(_load_from_subdirs(task_subdirs))

    duration_s = time.perf_counter() - start_time
    print(f"[lighteval.tasks] Loaded {len(loaded_configs)} task configs in {duration_s * 1000:.1f} ms")
    return loaded_configs


# Load all configs and add them to module namespace
_configs = _load_all_task_configs()
globals().update(_configs)

# Clean up
del _configs

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
import logging
from pathlib import Path

from lighteval.tasks.lighteval_task import LightevalTaskConfig


logger = logging.getLogger(__name__)


# Get the tasks directory
TASKS_DIR = Path(__file__).parent / "tasks"


def _load_all_task_configs():
    """Load all LightevalTaskConfig objects from all Python files in the tasks/ directory."""
    loaded_configs = {}

    # Get all Python files in the tasks directory (excluding __init__.py and subdirectories)
    task_files = [f for f in TASKS_DIR.glob("*.py") if f.name != "__init__.py"]

    for task_file in task_files:
        module_name = task_file.stem
        # Import the module
        module = importlib.import_module(f"lighteval.tasks.tasks.{module_name}")

        # Find all LightevalTaskConfig objects in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, LightevalTaskConfig):
                loaded_configs[attr_name] = attr

    return loaded_configs


# Load all configs and add them to module namespace
_configs = _load_all_task_configs()
globals().update(_configs)

# Clean up
del _configs

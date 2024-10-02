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

import collections
import importlib
import os
from itertools import groupby
from pathlib import Path
from pprint import pformat
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union

from datasets.load import dataset_module_factory

import lighteval.tasks.default_tasks as default_tasks
from lighteval.logging.hierarchical_logger import hlog, hlog_warn
from lighteval.tasks.extended import AVAILABLE_EXTENDED_TASKS_MODULES
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.utils.imports import CANNOT_USE_EXTENDED_TASKS_MSG, can_load_extended_tasks


# Helm, Bigbench, Harness are implementations following an evaluation suite setup
# Original follows the original implementation as closely as possible
# Leaderboard are the evaluations we fixed on the open llm leaderboard - you should get similar results
# Community are for community added evaluations
# Extended are for evaluations with custom logic
# Custom is for all the experiments you might want to do!
DEFAULT_SUITES = [
    "helm",
    "bigbench",
    "harness",
    "leaderboard",
    "lighteval",
    "original",
    "extended",
    "custom",
    "community",
]

TRUNCATE_FEW_SHOTS_DEFAULTS = True


class Registry:
    """
    The Registry class is used to manage the task registry and get task classes.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize the Registry class.

        Args:
            cache_dir (str): Directory path for caching.

        Attributes:
            cache_dir (str): Directory path for caching.
            TASK_REGISTRY (dict[str, LightevalTask]): A dictionary containing the registered tasks.
        """
        self.cache_dir: str = cache_dir
        self.TASK_REGISTRY: dict[str, LightevalTask] = {**create_config_tasks(cache_dir=cache_dir)}

    def get_task_class(
        self, task_name: str, custom_tasks_registry: Optional[dict[str, LightevalTask]] = None
    ) -> LightevalTask:
        """
        Get the task class based on the task name.

        Args:
            task_name (str): Name of the task.
            custom_tasks_registry (Optional[dict[str, LightevalTask]]): A dictionary containing custom tasks.

        Returns:
            LightevalTask: Task class.

        Raises:
            ValueError: If the task is not found in the task registry or custom task registry.
        """
        if task_name in self.TASK_REGISTRY:
            if custom_tasks_registry is not None and task_name in custom_tasks_registry:
                hlog_warn(
                    f"One of the tasks you requested ({task_name}) exists both in the default and custom tasks. Selecting the default task."
                )
            return self.TASK_REGISTRY[task_name]
        if custom_tasks_registry is not None and task_name in custom_tasks_registry:
            return custom_tasks_registry[task_name]
        hlog_warn(f"{task_name} not found in provided tasks")
        hlog_warn(pformat(self.TASK_REGISTRY))
        raise ValueError(
            f"Cannot find tasks {task_name} in task list or in custom task registry ({custom_tasks_registry})"
        )

    def get_task_dict(
        self, task_name_list: List[str], custom_tasks: Optional[Union[str, Path, ModuleType]] = None
    ) -> Dict[str, LightevalTask]:
        """
        Get a dictionary of tasks based on the task name list.

        Args:
            task_name_list (List[str]): A list of task names.
            custom_tasks (Optional[Union[str, ModuleType]]): Path to the custom tasks file or name of a module to import containing custom tasks or the module itself
            extended_tasks (Optional[str]): The path to the extended tasks group of submodules

        Returns:
            Dict[str, LightevalTask]: A dictionary containing the tasks.

        Notes:
            - If custom_tasks is provided, it will import the custom tasks module and create a custom tasks registry.
            - Each task in the task_name_list will be instantiated with the corresponding task class.
        """
        # Import custom tasks provided by the user
        custom_tasks_registry = None
        custom_tasks_module = []
        TASKS_TABLE = []
        if custom_tasks is not None:
            custom_tasks_module.append(create_custom_tasks_module(custom_tasks=custom_tasks))
        if can_load_extended_tasks():
            for extended_task_module in AVAILABLE_EXTENDED_TASKS_MODULES:
                custom_tasks_module.append(extended_task_module)
        else:
            hlog_warn(CANNOT_USE_EXTENDED_TASKS_MSG)

        for module in custom_tasks_module:
            TASKS_TABLE.extend(module.TASKS_TABLE)

        if len(TASKS_TABLE) > 0:
            custom_tasks_registry = create_config_tasks(meta_table=TASKS_TABLE, cache_dir=self.cache_dir)
            hlog(custom_tasks_registry)

        # Select relevant tasks given the subset asked for by the user
        tasks_dict = {}
        for task_name in task_name_list:
            task_class = self.get_task_class(task_name, custom_tasks_registry=custom_tasks_registry)
            tasks_dict[task_name] = task_class()

        return tasks_dict

    def print_all_tasks(self):
        """
        Print all the tasks in the task registry.
        """
        tasks_names = list(self.TASK_REGISTRY.keys())
        tasks_names.sort()
        for suite, g in groupby(tasks_names, lambda x: x.split("|")[0]):
            tasks_names = list(g)
            tasks_names.sort()
            print(f"\n- {suite}:")
            for task_name in tasks_names:
                print(f"  - {task_name}")


def create_custom_tasks_module(custom_tasks: Union[str, Path, ModuleType]) -> ModuleType:
    """Creates a custom task module to load tasks defined by the user in their own file.

    Args:
        custom_tasks (Optional[Union[str, ModuleType]]): Path to the custom tasks file or name of a module to import containing custom tasks or the module itself

    Returns:
        ModuleType: The newly imported/created custom tasks modules
    """
    if isinstance(custom_tasks, ModuleType):
        return custom_tasks
    if isinstance(custom_tasks, (str, Path)) and os.path.exists(custom_tasks):
        dataset_module = dataset_module_factory(str(custom_tasks))
        return importlib.import_module(dataset_module.module_path)
    if isinstance(custom_tasks, (str, Path)):
        return importlib.import_module(custom_tasks)
    raise ValueError(f"Cannot import custom tasks from {custom_tasks}")


def get_custom_tasks(custom_tasks: Union[str, ModuleType]) -> Tuple[ModuleType, str]:
    """Get all the custom tasks available from the given custom tasks file or module.

    Args:
        custom_tasks (Optional[Union[str, ModuleType]]): Path to the custom tasks file or name of a module to import containing custom tasks or the module itself
    """
    custom_tasks_module = create_custom_tasks_module(custom_tasks=custom_tasks)
    tasks_string = ""
    if hasattr(custom_tasks_module, "TASKS_GROUPS"):
        tasks_string = custom_tasks_module.TASKS_GROUPS
    return custom_tasks_module, tasks_string


def taskinfo_selector(
    tasks: str,
) -> tuple[list[str], dict[str, list[tuple[int, bool]]]]:
    """
    Converts a input string of tasks name to task information usable by lighteval.

    Args:
        tasks (str): A string containing a comma-separated list of tasks in the
            format "suite|task|few_shot|truncate_few_shots" or a path to a file
            containing a list of tasks.

    Returns:
        tuple[list[str], dict[str, list[tuple[int, bool]]]]: A tuple containing:
            - A sorted list of unique task names in the format "suite|task".
            - A dictionary mapping each task name to a list of tuples representing the few_shot and truncate_few_shots values.
    """
    few_shot_dict = collections.defaultdict(list)

    # We can provide a path to a file with a list of tasks
    if "." in tasks and os.path.exists(tasks):
        tasks = ",".join([line for line in open(tasks, "r").read().splitlines() if not line.startswith("#")])

    for task in tasks.split(","):
        try:
            suite_name, task_name, few_shot, truncate_few_shots = tuple(task.split("|"))
            truncate_few_shots = int(truncate_few_shots)
        except ValueError:
            raise ValueError(
                f"Cannot get task info from {task}. correct format is suite|task|few_shot|truncate_few_shots"
            )

        if truncate_few_shots not in [0, 1]:
            raise ValueError(f"TruncateFewShots must be 0 or 1, got {truncate_few_shots}")

        truncate_few_shots = bool(truncate_few_shots)
        few_shot = int(few_shot)

        if suite_name not in DEFAULT_SUITES:
            hlog(f"Suite {suite_name} unknown. This is not normal, unless you are testing adding new evaluations.")

        # Store few_shot info for each task name (suite|task)
        few_shot_dict[f"{suite_name}|{task_name}"].append((few_shot, truncate_few_shots))

    return sorted(few_shot_dict.keys()), {k: list(set(v)) for k, v in few_shot_dict.items()}


def create_config_tasks(
    meta_table: Optional[List[LightevalTaskConfig]] = None, cache_dir: Optional[str] = None
) -> Dict[str, LightevalTask]:
    """
    Create configuration tasks based on the provided meta_table.

    Args:
        meta_table: meta_table containing tasks
            configurations. If not provided, it will be loaded from TABLE_PATH.
        cache_dir: Directory to store cached data. If not
            provided, the default cache directory will be used.

    Returns:
        Dict[str, LightevalTask]: A dictionary of task names mapped to their corresponding LightevalTask classes.
    """

    def create_task(name, cfg: LightevalTaskConfig, cache_dir: str):
        class LightevalTaskFromConfig(LightevalTask):
            def __init__(self):
                super().__init__(name, cfg, cache_dir=cache_dir)

        return LightevalTaskFromConfig

    if meta_table is None:
        meta_table = [config for config in vars(default_tasks).values() if isinstance(config, LightevalTaskConfig)]

    tasks_with_config = {}
    # Every task is renamed suite|task, if the suite is in DEFAULT_SUITE
    for config in meta_table:
        if not any(suite in config.suite for suite in DEFAULT_SUITES):
            hlog_warn(
                f"This evaluation is not in any known suite: {config.name} is in {config.suite}, not in {DEFAULT_SUITES}. Skipping."
            )
            continue
        for suite in config.suite:
            if suite in DEFAULT_SUITES:
                tasks_with_config[f"{suite}|{config.name}"] = config

    return {task: create_task(task, cfg, cache_dir=cache_dir) for task, cfg in tasks_with_config.items()}

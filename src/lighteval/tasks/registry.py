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
import copy
import importlib
import logging
import os
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from types import ModuleType

from datasets.load import dataset_module_factory

import lighteval.tasks.default_tasks as default_tasks
from lighteval.tasks.extended import AVAILABLE_EXTENDED_TASKS_MODULES
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.utils.imports import CANNOT_USE_EXTENDED_TASKS_MSG, can_load_extended_tasks


logger = logging.getLogger(__name__)

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
    "test",
]

TRUNCATE_FEW_SHOTS_DEFAULTS = True


class Registry:
    """
    The Registry class is used to manage the task registry and get task classes.
    """

    def __init__(self, custom_tasks: str | Path | ModuleType | None = None):
        """
        Initialize the Registry class.
        Registry is responsible for holding a dict of task and their config, initializing a LightevalTask instance when asked.

        Args:
            custom_tasks (Optional[Union[str, Path, ModuleType]]): Custom tasks to be included in registry. Can be a string path, Path object, or a module.
                Each custom task should be a module with a TASKS_TABLE exposing a list of LightevalTaskConfig.
                E.g:
                TASKS_TABLE = [
                    LightevalTaskConfig(
                        name="custom_task",
                        suite="custom",
                        ...
                    )
                ]
        """
        self._custom_tasks = custom_tasks

    def get_tasks_from_configs(self, task_configs: list[LightevalTaskConfig]) -> dict[str, LightevalTask]:
        return {f"{config.full_name}": LightevalTask(config=config) for config in task_configs}

    def get_tasks_configs(self, task: str) -> list[LightevalTaskConfig]:
        """
        task is a string of the form "suite|task|few_shot|truncate_few_shots,suite|task|few_shot|truncate_few_shots"

        returns a LightevalTaskConfig object based on the task name and fewshot and truncate_few_shots values.
        """
        task_info_dict = self.taskinfo_selector(task)
        configs = []

        for task_name, task_info in task_info_dict.items():
            # We can have multiple few_shot and truncate_few_shots values for the same task
            for task_info_dict in task_info:
                config = self.task_registry.get(task_name)
                if config is not None:
                    config = copy.deepcopy(config)
                    config.num_fewshots = task_info_dict["fewshots"]
                    config.truncate_fewshots = task_info_dict["truncate_fewshots"]
                    config.full_name = f"{task_name}|{config.num_fewshots}"
                    configs.append(config)
                else:
                    raise ValueError(f"Cannot find task {task_name} in task list or in custom task registry")

        return configs

    @property
    @lru_cache
    def task_registry(self) -> dict[str, LightevalTaskConfig]:
        """
        Returns:
            dict[str, LazyLightevalTask]: A dictionary mapping task names (suite|task) to their corresponding LightevalTask classes.

        Example:
            {
                "lighteval|arc_easy": LightevalTaskConfig(name="arc_easy", suite="lighteval", ...),
            }
        """
        custom_tasks_registry = {}
        custom_tasks_module = []
        custom_task_configs = []

        if self._custom_tasks is not None:
            custom_tasks_module.append(Registry.create_custom_tasks_module(custom_tasks=self._custom_tasks))
        if can_load_extended_tasks():
            for extended_task_module in AVAILABLE_EXTENDED_TASKS_MODULES:
                custom_tasks_module.append(extended_task_module)
        else:
            logger.warning(CANNOT_USE_EXTENDED_TASKS_MSG)

        for module in custom_tasks_module:
            custom_task_configs.extend(module.TASKS_TABLE)
            logger.info(f"Found {len(module.TASKS_TABLE)} custom tasks in {module.__file__}")

        if len(custom_task_configs) > 0:
            custom_tasks_registry = Registry.create_task_config_dict(meta_table=custom_task_configs)

        default_tasks_registry = Registry.create_task_config_dict()

        # Check the overlap between default_tasks_registry and custom_tasks_registry
        intersection = set(default_tasks_registry.keys()).intersection(set(custom_tasks_registry.keys()))
        if len(intersection) > 0:
            logger.warning(
                f"Following tasks ({intersection}) exists both in the default and custom tasks. Will use the custom ones on conflict."
            )

        return {**default_tasks_registry, **custom_tasks_registry}

    def taskinfo_selector(self, tasks: str) -> dict[str, list[dict]]:
        """
        Converts a input string of tasks name to task information usable by lighteval.

        Args:
            tasks (str): A string containing a comma-separated list of tasks definitions in the
                format: "task_definition", where it can be
                containing a list of tasks.
                where task_definition can be:
                - path to a file containing a list of tasks (one per line)
                - task group defined in TASKS_GROUPS dict in custom tasks file
                - task name with few shot in format "suite|task|few_shot|truncate_few_shots"
                - task superset in format "suite|task_superset|few_shot|truncate_few_shots" (superset will run all tasks with format "suite|task_superset:{subset}|few_shot|truncate_few_shots")


        Returns:
            tuple[list[str], dict[str, list[tuple[int, bool]]]]: A tuple containing:
                - A sorted list of unique task names in the format "suite|task".
                - A dictionary mapping each task name to a list of tuples representing the few_shot and truncate_few_shots values.
        """
        few_shot_dict = collections.defaultdict(list)

        # We can provide a path to a file with a list of tasks or a string of comma-separated tasks
        if os.path.exists(tasks):
            with open(tasks, "r") as f:
                tasks_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            tasks_list = tasks.split(",")

        # At this point the strings are either task name/superset name or group names
        # Here we deal with group names and map them to corresponding tasks
        expanded_tasks_list: list[str] = []
        for maybe_task_group in tasks_list:
            # We either expand the group (in case it's a group name), or we keep it as is (in case it's a task name or superset name)
            expanded_tasks = self.task_groups_dict.get(maybe_task_group, [maybe_task_group])
            if len(expanded_tasks) > 1:
                logger.info(f"Expanding task group {maybe_task_group} to {expanded_tasks}")
            expanded_tasks_list.extend(expanded_tasks)

        for task in expanded_tasks_list:
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
                logger.warning(
                    f"Suite {suite_name} unknown. This is not normal, unless you are testing adding new evaluations."
                )

            # This adds support for task supersets (eg: mmlu -> all the mmlu tasks)
            for expanded_task in self.expand_task_definition(f"{suite_name}|{task_name}"):
                # Store few_shot info for each task name (suite|task)
                few_shot_dict[expanded_task].append({"fewshots": few_shot, "truncate_fewshots": truncate_few_shots})

        return few_shot_dict

    @property
    @lru_cache
    def _task_superset_dict(self):
        """
        Returns:
            dict[str, list[str]]: A dictionary where keys are task super set names (suite|task) and values are lists of task subset names (suite|task).

        Example:
            {
                "lighteval|mmlu" -> ["lighteval|mmlu:abstract_algebra", "lighteval|mmlu:college_biology", ...]
            }
        """
        # Note: sorted before groupby is important as the python implementation of groupby does not
        # behave like sql groupby. For more info see the docs of itertools.groupby
        superset_dict = {k: list(v) for k, v in groupby(sorted(self.task_registry.keys()), lambda x: x.split(":")[0])}
        # Only consider supersets with more than one task
        return {k: v for k, v in superset_dict.items() if len(v) > 1}

    @property
    @lru_cache
    def task_groups_dict(self) -> dict[str, list[str]]:
        """
        Returns:
            dict[str, list[str]]: A dictionary where keys are task group names and values are lists of task names (suite|task).

        Example:
            {
                "all_custom": ["custom|task1", "custom|task2", "custom|task3"],
                "group1": ["custom|task1", "custom|task2"],
            }
        """
        if self._custom_tasks is None:
            return {}
        custom_tasks_module = Registry.create_custom_tasks_module(custom_tasks=self._custom_tasks)
        tasks_group_dict = {}
        if hasattr(custom_tasks_module, "TASKS_GROUPS"):
            tasks_group_dict = custom_tasks_module.TASKS_GROUPS

        # We should allow defining task groups as comma-separated strings or lists of tasks
        return {k: v if isinstance(v, list) else v.split(",") for k, v in tasks_group_dict.items()}

    def expand_task_definition(self, task_definition: str):
        """
        Args:
            task_definition (str): Task definition to expand. In format:
                - suite|task
                - suite|task_superset (e.g lighteval|mmlu, which runs all the mmlu subtasks)
        Returns:
            list[str]: List of task names (suite|task)
        """

        # Try if it's a task superset
        tasks = self._task_superset_dict.get(task_definition, None)
        if tasks is not None:
            return tasks

        # Then it must be a single task
        return [task_definition]

    def print_all_tasks(self):
        """
        Print all the tasks in the task registry.
        """
        tasks_names = list(self.task_registry.keys())
        tasks_names.sort()
        for suite, g in groupby(tasks_names, lambda x: x.split("|")[0]):
            tasks_names = list(g)
            tasks_names.sort()
            print(f"\n- {suite}:")
            for task_name in tasks_names:
                print(f"  - {task_name}")

    @staticmethod
    def create_custom_tasks_module(custom_tasks: str | Path | ModuleType) -> ModuleType:
        """Creates a custom task module to load tasks defined by the user in their own file.

        Args:
            custom_tasks (Optional[Union[str, ModuleType]]): Path to the custom tasks file or name of a module to import containing custom tasks or the module itself

        Returns:
            ModuleType: The newly imported/created custom tasks modules
        """
        if isinstance(custom_tasks, ModuleType):
            return custom_tasks
        if isinstance(custom_tasks, (str, Path)) and os.path.exists(custom_tasks):
            dataset_module = dataset_module_factory(str(custom_tasks), trust_remote_code=True)
            return importlib.import_module(dataset_module.module_path)
        if isinstance(custom_tasks, (str, Path)):
            return importlib.import_module(str(custom_tasks))

    @staticmethod
    def create_task_config_dict(meta_table: list[LightevalTaskConfig] | None = None) -> dict[str, LightevalTaskConfig]:
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

        if meta_table is None:
            meta_table = [config for config in vars(default_tasks).values() if isinstance(config, LightevalTaskConfig)]

        tasks_with_config: dict[str, LightevalTaskConfig] = {}
        for config in meta_table:
            if not any(suite in config.suite for suite in DEFAULT_SUITES):
                logger.warning(
                    f"This evaluation is not in any known suite: {config.name} is in {config.suite}, not in {DEFAULT_SUITES}. Skipping."
                )
                continue
            for suite in config.suite:
                if suite in DEFAULT_SUITES:
                    tasks_with_config[f"{suite}|{config.name}"] = config

        return tasks_with_config

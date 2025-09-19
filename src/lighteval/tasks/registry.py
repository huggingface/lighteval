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

import ast
import collections
import copy
import importlib
import importlib.util
import logging
import os
import sys
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from types import ModuleType

import lighteval.tasks.default_tasks as default_tasks
from lighteval.tasks.extended import AVAILABLE_EXTENDED_TASKS_MODULES
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig


# Import community tasks
AVAILABLE_COMMUNITY_TASKS_MODULES = []


def load_community_tasks():
    """Dynamically load community tasks, handling errors gracefully.

    Returns:
        list: List of successfully loaded community task modules
    """
    modules = []
    try:
        # Community tasks are in the lighteval directory, not under src
        community_path = Path(__file__).parent.parent.parent.parent / "community_tasks"
        if not community_path.exists():
            return modules

        # Ensure the parent directory is on sys.path so we can import `community_tasks.*`
        parent_dir = str(community_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # List all python files in community_tasks
        community_files = [p.stem for p in community_path.glob("*.py") if not p.name.startswith("_")]

        for module_name in community_files:
            try:
                module = importlib.import_module(f"community_tasks.{module_name}")
                if hasattr(module, "TASKS_TABLE"):
                    modules.append(module)
                    logger.info(f"Successfully loaded community tasks from {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load community tasks from {module_name}: {e}")
    except Exception as e:
        logger.warning(f"Error loading community tasks directory: {e}")

    return modules


logger = logging.getLogger(__name__)

# Helm, Bigbench, Harness are implementations following an evaluation suite setup
# Original follows the original implementation as closely as possible
# Leaderboard are the evaluations we fixed on the open llm leaderboard - you should get similar results
# Community are for community added evaluations
# Extended are for evaluations with custom logic
# Custom is for all the experiments you might want to do!

# Core suites - always available without extra dependencies
CORE_SUITES = [
    "helm",
    "bigbench",
    "harness",
    "leaderboard",
    "lighteval",
    "original",
    "extended",
    "custom",
    "test",
]

# Optional suites - may require extra dependencies
OPTIONAL_SUITES = [
    "community",
    "multilingual",
]

DEFAULT_SUITES = CORE_SUITES + OPTIONAL_SUITES


class Registry:
    """The Registry class is used to manage the task registry and get task classes."""

    def __init__(
        self,
        tasks: str | Path | None = None,
        custom_tasks: str | Path | ModuleType | None = None,
        load_community: bool = False,
        load_extended: bool = False,
        load_multilingual: bool = False,
    ):
        """
        Initialize the Registry class.
        Registry is responsible for holding a dict of task and their config, initializing a LightevalTask instance when asked.

        Args:
            tasks: Task specification string or path to file containing task list.
            custom_tasks: Custom tasks to be included in the registry. Can be:
                - A string path to a Python file containing custom tasks
                - A Path object pointing to a custom tasks file
                - A module object containing custom task configurations
                - None for default behavior (no custom tasks)
            load_community: Whether to load community-contributed tasks.
            load_extended: Whether to load extended tasks with custom logic.
            load_multilingual: Whether to load multilingual tasks.

                Each custom task module should contain a TASKS_TABLE exposing
                a list of LightevalTaskConfig objects.

        Example:
                    TASKS_TABLE = [
                        LightevalTaskConfig(
                            name="custom_task",
                            suite="custom",
                            ...
                        )
                    ]
        """
        self._custom_tasks = custom_tasks

        if tasks is None:
            logger.warning(
                "You passed no task name. This should only occur if you are using the CLI to inspect tasks."
            )
            self.tasks_list = []
        else:
            self.tasks_list = self._get_full_task_list_from_input_string(tasks)
        # These parameters are dynamically set by the task names provided, thanks to `activate_suites_to_load`,
        # except in the `tasks` CLI command to display the full list
        self._load_community = load_community
        self._load_extended = load_extended
        self._load_multilingual = load_multilingual
        self._activate_loading_of_optional_suite()  # we dynamically set the loading parameters

        # We load all task to
        self._task_registry = self._load_full_registry()

        self.task_to_configs = self._update_task_configs()

    def _get_full_task_list_from_input_string(self, tasks: str | Path) -> list[str]:
        """Converts an input string (either a path to file with a list of tasks or a string of comma-separated tasks) into an actual list"""
        if os.path.exists(tasks):
            with open(tasks, "r") as f:
                tasks_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            tasks_list = tasks.split(",")

        # We might have tasks provided as task groups in the custom tasks
        # We load the whole task_groups mapping
        if self._custom_tasks is None:
            task_groups = {}
        else:
            custom_tasks_module = Registry.create_custom_tasks_module(custom_tasks=self._custom_tasks)
            tasks_group_dict = {}
            if hasattr(custom_tasks_module, "TASKS_GROUPS"):
                tasks_group_dict = custom_tasks_module.TASKS_GROUPS

            # We should allow defining task groups as comma-separated strings or lists of tasks
            task_groups = {k: v if isinstance(v, list) else v.split(",") for k, v in tasks_group_dict.items()}

        # Then link actual task_group to task list if needed
        # (At this point the strings are either task name/superset name or group names)
        expanded_tasks_list: list[str] = []
        for maybe_task_group in tasks_list:
            # We either expand the group (in case it's a group name), or we keep it as is (in case it's a task name or superset name)
            expanded_tasks = task_groups.get(maybe_task_group, [maybe_task_group])
            if len(expanded_tasks) > 1:
                logger.info(f"Expanding task group {maybe_task_group} to {expanded_tasks}")
            expanded_tasks_list.extend(expanded_tasks)

        # We remove exact duplicates
        expanded_tasks_list = list(set(expanded_tasks_list))

        return expanded_tasks_list

    def _activate_loading_of_optional_suite(self) -> None:
        """Dynamically selects which of the optional suite we want to load."""
        suites = {task.split("|")[0] for task in self.tasks_list}

        for suite_name in suites:
            if suite_name not in DEFAULT_SUITES:
                logger.warning(
                    f"Suite {suite_name} unknown. This is not normal, unless you are testing adding new evaluations."
                )

        if "extended" in suites:
            self._load_extended = True
        if "multilingual" in suites:
            self._load_multilingual = True
        if "community" in suites:
            self._load_community = True

    def _load_full_registry(self) -> dict[str, LightevalTaskConfig]:
        """
        Returns:
            dict[str, LightevalTaskConfig]: A dictionary mapping task names (suite|task) to their corresponding LightevalTask classes.

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

        # Need to load extended tasks
        if self._load_extended:
            for extended_task_module in AVAILABLE_EXTENDED_TASKS_MODULES:
                custom_tasks_module.append(extended_task_module)

        # Need to load community tasks
        if self._load_community:
            community_modules = load_community_tasks()
            for community_task_module in community_modules:
                custom_tasks_module.append(community_task_module)

        # Need to load multilingual tasks
        if self._load_multilingual:
            import lighteval.tasks.multilingual.tasks as multilingual_tasks

            custom_tasks_module.append(multilingual_tasks)

        # We load all
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

    def _update_task_configs(self) -> dict[str, LightevalTaskConfig]:  # noqa: C901
        """
        Updates each config depending on the input tasks (we replace all provided params, like few shot number, sampling params, etc)
        """
        task_to_configs = collections.defaultdict(list)

        # We map all tasks to their parameters
        for task in self.tasks_list:
            metric_params_dict = {}
            try:
                if task.count("|") == 3:
                    logger.warning(
                        "Deprecation warning: You provided 4 arguments in your task name, but we no longer support the `truncate_fewshot` option. We will ignore the parameter for now, but it will fail in a couple of versions, so you should change your task name to `suite|task|num_fewshot`."
                    )
                    suite_name, task_name, few_shot, _ = tuple(task.split("|"))
                else:
                    suite_name, task_name, few_shot = tuple(task.split("|"))
                if "@" in task_name:
                    split_task_name = task_name.split("@")
                    task_name, metric_params = split_task_name[0], split_task_name[1:]
                    # We convert k:v to {"k": "v"}, then to correct type
                    metric_params_dict = dict(item.split("=") for item in metric_params if item)
                    metric_params_dict = {k: ast.literal_eval(v) for k, v in metric_params_dict.items()}
                few_shot = int(few_shot)

            except ValueError:
                raise ValueError(f"Cannot get task info from {task}. correct format is suite|task|few_shot")

            # This adds support for task supersets (eg: mmlu -> all the mmlu tasks)
            for expanded_task in self._expand_task_definition(f"{suite_name}|{task_name}"):
                # todo: it's likely we'll want this step at the list set up step, not here

                # We load each config
                config = self._task_registry.get(expanded_task)
                if config is None:
                    raise ValueError(f"Cannot find task {expanded_task} in task list or in custom task registry")

                config = copy.deepcopy(config)
                config.num_fewshots = few_shot
                config.full_name = f"{expanded_task}|{config.num_fewshots}"
                # If some tasks are parametrizable and in cli, we set attributes here
                for metric in [m for m in config.metrics if "@" in m.metric_name]:  # parametrizable metric
                    for attribute, value in metric_params_dict.items():
                        setattr(metric.sample_level_fn, attribute, value)
                    required = getattr(metric.sample_level_fn, "attribute_must_be_set", [])
                    for attribute in required:
                        if getattr(metric.sample_level_fn, attribute) is None:
                            raise ValueError(
                                f"Metric {metric.metric_name} for task {expanded_task} "
                                f"was not correctly parametrized. Forgot to set '{attribute}'."
                            )

                task_to_configs[expanded_task].append(config)

        return task_to_configs

    def load_tasks(self) -> dict[str, LightevalTask]:
        if len(self.task_to_configs) == 0:  # we're in cli to analyse tasks, we return all tasks
            return {f"{config.full_name}": LightevalTask(config=config) for config in self._task_registry.values()}

        # We return only the tasks of interest
        return {
            f"{config.full_name}": LightevalTask(config=config)
            for configs in self.task_to_configs.values()
            for config in configs
        }

    @property
    @lru_cache
    def _task_superset_dict(self):
        """Returns:
            dict[str, list[str]]: A dictionary where keys are task super set names (suite|task) and values are lists of task subset names (suite|task).

        Example:
            {
                "lighteval|mmlu" -> ["lighteval|mmlu:abstract_algebra", "lighteval|mmlu:college_biology", ...]
            }
        """
        # Note: sorted before groupby is important as the python implementation of groupby does not
        # behave like sql groupby. For more info see the docs of itertools.groupby
        superset_dict = {k: list(v) for k, v in groupby(sorted(self._task_registry.keys()), lambda x: x.split(":")[0])}
        # Only consider supersets with more than one task
        return {k: v for k, v in superset_dict.items() if len(v) > 1}

    def _expand_task_definition(self, task_definition: str):
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
            module_name = os.path.splitext(os.path.basename(custom_tasks))[0]
            spec = importlib.util.spec_from_file_location(module_name, custom_tasks)

            if spec is None:
                raise ValueError(f"Cannot find module {module_name} at {custom_tasks}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        if isinstance(custom_tasks, (str, Path)):
            return importlib.import_module(str(custom_tasks))

    @staticmethod
    def create_task_config_dict(meta_table: list[LightevalTaskConfig] | None = None) -> dict[str, LightevalTaskConfig]:
        """Create configuration tasks based on the provided meta_table.

        Args:
            meta_table: meta_table containing tasks
                configurations. If not provided, it will be loaded from TABLE_PATH.

        Returns:
            Dict[str, LightevalTaskConfig]: A dictionary of task names mapped to their corresponding LightevalTaskConfig.
        """
        if meta_table is None:
            meta_table = [config for config in vars(default_tasks).values() if isinstance(config, LightevalTaskConfig)]

        tasks_with_config: dict[str, LightevalTaskConfig] = {}
        for config in meta_table:
            for suite in config.suite:
                if suite in DEFAULT_SUITES:
                    tasks_with_config[f"{suite}|{config.name}"] = config

        return tasks_with_config

    def print_all_tasks(self, suites: str | None = None):
        """Print all the tasks in the task registry.

        Args:
            suites: Comma-separated list of suites to display. If None, shows core suites only.
                   Use 'all' to show all available suites (core + optional).
                   Special handling for 'multilingual' suite with dependency checking.
        """
        # Parse requested suites
        if suites is None:
            requested_suites = CORE_SUITES.copy()
        else:
            requested_suites = [s.strip() for s in suites.split(",")]

            # Handle 'all' special case
            if "all" in requested_suites:
                requested_suites = DEFAULT_SUITES.copy()

            # Check for multilingual dependencies if requested
            if "multilingual" in requested_suites:
                import importlib.util

                if importlib.util.find_spec("langcodes") is None:
                    logger.warning(
                        "Multilingual tasks require additional dependencies (langcodes). "
                        "Install them with: pip install langcodes"
                    )
                    requested_suites.remove("multilingual")

        # Get all tasks and filter by requested suites
        all_tasks = list(self._task_registry.keys())
        tasks_names = [task for task in all_tasks if task.split("|")[0] in requested_suites]

        # Ensure all requested suites are present (even if empty)
        suites_in_registry = {name.split("|")[0] for name in tasks_names}
        for suite in requested_suites:
            if suite not in suites_in_registry:
                # We add a dummy task to make sure the suite is printed
                tasks_names.append(f"{suite}|")

        tasks_names.sort()

        print(f"Displaying tasks for suites: {', '.join(requested_suites)}")
        print("=" * 60)

        for suite, g in groupby(tasks_names, lambda x: x.split("|")[0]):
            tasks_in_suite = [name for name in g if name.split("|")[1]]  # Filter out dummy tasks
            tasks_in_suite.sort()

            print(f"\n- {suite}:")
            if not tasks_in_suite:
                print("  (no tasks in this suite)")
            else:
                for task_name in tasks_in_suite:
                    print(f"  - {task_name}")

        # Print summary
        total_tasks = len([t for t in tasks_names if t.split("|")[1]])
        print(f"\nTotal tasks displayed: {total_tasks}")

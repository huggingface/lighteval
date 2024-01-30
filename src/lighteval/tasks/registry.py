import collections
import importlib
import os
from pprint import pformat
from types import ModuleType
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from datasets.load import dataset_module_factory

from lighteval.logging.hierarchical_logger import hlog, hlog_warn
from lighteval.tasks.lighteval_task import LightevalTask


# original is the reimplementation of original evals
# custom is to play around
DEFAULT_SUITES = ["helm", "bigbench", "lighteval", "original", "custom"]

TRUNCATE_FEW_SHOTS_DEFAULTS = True

TABLE_PATH = os.path.join(os.path.dirname(__file__), "tasks_table.jsonl")


class Registry:
    """
    The Registry class is used to manage the task registry and get task classes.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize the Registry class.

        Args:
            cache_dir (str): The directory path for caching.

        Attributes:
            cache_dir (str): The directory path for caching.
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
            task_name (str): The name of the task.
            custom_tasks_registry (Optional[dict[str, LightevalTask]]): A dictionary containing custom tasks.

        Returns:
            LightevalTask: The task class.

        Raises:
            ValueError: If the task is not found in the task registry or custom task registry.
        """
        if task_name in self.TASK_REGISTRY:
            return self.TASK_REGISTRY[task_name]
        elif custom_tasks_registry is not None and task_name in custom_tasks_registry:
            return custom_tasks_registry[task_name]
        else:
            hlog_warn(f"{task_name} not found in provided tasks")
            hlog_warn(pformat(self.TASK_REGISTRY))
            raise ValueError(
                f"Cannot find tasks {task_name} in task list or in custom task registry ({custom_tasks_registry})"
            )

    def get_task_dict(
        self, task_name_list: List[str], custom_tasks_file: Optional[str] = None
    ) -> Dict[str, LightevalTask]:
        """
        Get a dictionary of tasks based on the task name list.

        Args:
            task_name_list (List[str]): A list of task names.
            custom_tasks_file (Optional[str]): The path to the custom tasks file.

        Returns:
            Dict[str, LightevalTask]: A dictionary containing the tasks.

        Notes:
            - If custom_tasks_file is provided, it will import the custom tasks module and create a custom tasks registry.
            - Each task in the task_name_list will be instantiated with the corresponding task class.
        """
        if custom_tasks_file is not None:
            dataset_module = dataset_module_factory(str(custom_tasks_file))
            custom_tasks_module = importlib.import_module(dataset_module.module_path)
            custom_tasks_registry = create_config_tasks(
                meta_table=custom_tasks_module.TASKS_TABLE, cache_dir=self.cache_dir
            )
            hlog(custom_tasks_registry)
        else:
            custom_tasks_module = None
            custom_tasks_registry = None

        tasks_dict = {}
        for task_name in task_name_list:
            task_class = self.get_task_class(task_name, custom_tasks_registry=custom_tasks_registry)
            tasks_dict[task_name] = task_class(custom_tasks_module=custom_tasks_module)

        return tasks_dict


def get_custom_tasks(custom_tasks_file: str) -> Tuple[ModuleType, str]:
    dataset_module = dataset_module_factory(str(custom_tasks_file))
    custom_tasks_module = importlib.import_module(dataset_module.module_path)
    tasks_string = ""
    if hasattr(custom_tasks_module, "TASKS_GROUPS"):
        tasks_string = custom_tasks_module.TASKS_GROUPS
    return custom_tasks_module, tasks_string


def taskinfo_selector(
    tasks: str,
) -> tuple[list[str], dict[str, list[tuple[int, bool]]]]:
    """
    Selects task information based on the given tasks and description dictionary path.

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
    meta_table: Optional[Dataset] = None, cache_dir: Optional[str] = None
) -> Dict[str, LightevalTask]:
    """
    Create configuration tasks based on the provided meta_table.

    Args:
        meta_table (Optional[Dataset]): The meta_table containing task
            configurations. If not provided, it will be loaded from TABLE_PATH.
        cache_dir (Optional[str]): The directory to store cached data. If not
            provided, the default cache directory will be used.

    Returns:
        Dict[str, LightevalTask]: A dictionary of task names mapped to their corresponding LightevalTask classes.
    """

    def create_task(name, cfg, cache_dir):
        class LightevalTaskFromConfig(LightevalTask):
            def __init__(self, custom_tasks_module=None):
                super().__init__(name, cfg, cache_dir=cache_dir, custom_tasks_module=custom_tasks_module)

        return LightevalTaskFromConfig

    if meta_table is None:
        meta_table = Dataset.from_json(TABLE_PATH)

    tasks_with_config = {}
    # Every task is renamed suite|task, if the suite is in DEFAULT_SUITE
    for line in meta_table:
        if not any(suite in line["suite"] for suite in DEFAULT_SUITES):
            hlog_warn(
                f"This evaluation is not in any known suite: {line['name']} is in {line['suite']}, not in {DEFAULT_SUITES}. Skipping."
            )
            continue
        for suite in line["suite"]:
            if suite in DEFAULT_SUITES:
                tasks_with_config[f"{suite}|{line['name']}"] = line

    return {task: create_task(task, cfg, cache_dir=cache_dir) for task, cfg in tasks_with_config.items()}


def task_to_suites(suites_selection: list = None):
    task_to_suites = {}
    meta_table = Dataset.from_json(TABLE_PATH)
    for line in meta_table:
        if suites_selection is None:
            task_to_suites[line["name"]] = line["suite"]
        else:
            task_to_suites[line["name"]] = [suite for suite in line["suite"] if suite in suites_selection]

    return task_to_suites

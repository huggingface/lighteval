import inspect
import logging
import types
from typing import List

from lighteval.tasks.lighteval_task import LightevalTaskConfig


logger = logging.getLogger(__name__)


def _validate_base_types(config: LightevalTaskConfig) -> List[str]:
    errors = []
    if not isinstance(config.name, str) or not config.name:
        errors.append(f"Task name must be a non-empty string. Got: {config.name}")
    if not isinstance(config.hf_repo, str) or not config.hf_repo:
        errors.append(f"Hugging Face repo (hf_repo) must be a non-empty string. Got: {config.hf_repo}")
    if not config.metrics or len(config.metrics) == 0:
        errors.append("Task config must define at least one metric in the 'metrics' list.")
    return errors


def _validate_splits(config: LightevalTaskConfig) -> List[str]:
    errors = []
    avail_splits = set(config.hf_avail_splits) if config.hf_avail_splits else set()
    for eval_split in config.evaluation_splits:
        if eval_split not in avail_splits:
            errors.append(f"evaluation_split '{eval_split}' is not declared in hf_avail_splits {list(avail_splits)}.")
    if config.few_shots_split is not None:
        if config.few_shots_split not in avail_splits:
            errors.append(
                f"few_shots_split '{config.few_shots_split}' is not declared in hf_avail_splits {list(avail_splits)}."
            )
    return errors


def _validate_prompt_function(config: LightevalTaskConfig) -> List[str]:
    errors = []
    if not callable(config.prompt_function):
        errors.append(f"prompt_function must be a callable (function). Got: {type(config.prompt_function)}")
        return errors

    try:
        sig = inspect.signature(config.prompt_function)
        if len(sig.parameters) < 1:
            errors.append("prompt_function must accept at least one parameter (the dataset row dict).")
        if sig.return_annotation is not inspect.Signature.empty:
            if getattr(sig.return_annotation, "__name__", str(sig.return_annotation)) not in [
                "Doc",
                "lighteval.tasks.requests.Doc",
            ]:
                logger.warning(f"prompt_function return annotation is {sig.return_annotation}, expected 'Doc'.")
    except ValueError:
        pass
    return errors


def validate_task_config(config: LightevalTaskConfig) -> List[str]:
    """
    Performs strict pure-Python static validation on a LightevalTaskConfig.
    Returns a list of error strings. If the list is empty, the config is structurally valid.
    """
    errors: List[str] = []
    errors.extend(_validate_base_types(config))
    errors.extend(_validate_splits(config))
    errors.extend(_validate_prompt_function(config))
    return errors


def validate_task_module(module: types.ModuleType) -> List[str]:
    """
    Validates the module boundary to ensure it correctly exports tasks for the registry.
    This prevents silent failures where a config is written but not correctly exported.
    """
    errors: List[str] = []

    if not hasattr(module, "TASKS_TABLE"):
        errors.append("Module is missing the required 'TASKS_TABLE' export list.")
        return errors

    tasks_table = getattr(module, "TASKS_TABLE")
    if not isinstance(tasks_table, list):
        errors.append(f"'TASKS_TABLE' must be a list. Got: {type(tasks_table)}")
        return errors

    if len(tasks_table) == 0:
        errors.append("'TASKS_TABLE' is empty. At least one task must be exported.")

    for idx, task_config in enumerate(tasks_table):
        if not isinstance(task_config, LightevalTaskConfig):
            errors.append(f"Item at index {idx} in 'TASKS_TABLE' is not a LightevalTaskConfig object.")
            continue

        config_errors = validate_task_config(task_config)
        for err in config_errors:
            errors.append(f"[Task: {getattr(task_config, 'name', 'Unknown')}] {err}")

    return errors

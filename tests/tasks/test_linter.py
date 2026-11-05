import types

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.linter import validate_task_module
from lighteval.tasks.requests import Doc


def test_linter_catches_missing_tasks_table():
    # Simulate a user who wrote the config but forgot the TASKS_TABLE export
    mock_module = types.ModuleType("mock_custom_task")

    errors = validate_task_module(mock_module)
    assert len(errors) == 1
    assert "missing the required 'TASKS_TABLE'" in errors[0]


def test_linter_catches_invalid_tasks_table_type():
    # Simulate a user who incorrectly defined TASKS_TABLE as a dict instead of a list
    mock_module = types.ModuleType("mock_custom_task")
    mock_module.TASKS_TABLE = {"my_task_name": "config_object"}

    errors = validate_task_module(mock_module)
    assert len(errors) == 1
    assert "'TASKS_TABLE' must be a list" in errors[0]


def test_linter_catches_invalid_config_inside_table():
    # Simulate a user who exported the list, but it contains a string instead of a LightevalTaskConfig
    mock_module = types.ModuleType("mock_custom_task")
    mock_module.TASKS_TABLE = ["This is not a LightevalTaskConfig"]

    errors = validate_task_module(mock_module)
    assert len(errors) == 1
    assert "is not a LightevalTaskConfig object" in errors[0]


def test_linter_validates_perfect_module_export():
    # Simulate a flawless, production-ready module export
    def mock_prompt_fn(line: dict) -> Doc:
        return Doc(task_name="test", query="q", choices=[], instruction="", target_for_fewshot_context="a")

    perfect_config = LightevalTaskConfig(
        name="test_task",
        hf_repo="huggingface/mock_repo",
        hf_subset="default",
        metrics=[Metrics.exact_match],
        prompt_function=mock_prompt_fn,
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
    )

    mock_module = types.ModuleType("mock_custom_task")
    mock_module.TASKS_TABLE = [perfect_config]

    errors = validate_task_module(mock_module)
    # The linter should find zero structural or namespace errors
    assert len(errors) == 0


def test_linter_catches_split_mismatch_inside_module():
    # Simulate a user requesting a split that doesn't exist on the Hugging Face repo
    def mock_prompt_fn(line: dict) -> Doc:
        return Doc(task_name="test", query="q", choices=[], instruction="", target_for_fewshot_context="a")

    broken_config = LightevalTaskConfig(
        name="broken_task",
        hf_repo="huggingface/mock_repo",
        hf_subset="default",
        metrics=[Metrics.exact_match],
        prompt_function=mock_prompt_fn,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["validation"],  # "validation" is not in hf_avail_splits!
    )

    mock_module = types.ModuleType("mock_custom_task")
    mock_module.TASKS_TABLE = [broken_config]

    errors = validate_task_module(mock_module)
    assert len(errors) == 1
    assert "evaluation_split 'validation' is not declared in hf_avail_splits" in errors[0]

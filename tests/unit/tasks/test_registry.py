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

import pytest

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry


TASKS_TABLE = [
    LightevalTaskConfig(
        name="test_task_revision",
        # Won't be called, so it can be anything
        prompt_function=lambda x: x,  # type: ignore
        hf_repo="test",
        hf_subset="default",
        evaluation_splits=["train"],
        metrics=[],
    )
]

TASKS_GROUPS = {
    "zero_and_one": "custom|test_task_revision|0,custom|test_task_revision|1",
    "all_mmlu": "original|mmlu|3",
}


def test_custom_task_groups():
    """
    Tests that task info selector correctly handles custom task groups.
    """
    registry = Registry(tasks="zero_and_one", custom_tasks="tests.unit.tasks.test_registry")

    assert set(registry.tasks_list) == {"custom|test_task_revision|0", "custom|test_task_revision|1"}

    assert set(registry.task_to_configs.keys()) == {"custom|test_task_revision"}

    task_info: list[LightevalTaskConfig] = registry.task_to_configs["custom|test_task_revision"]
    assert {task_info[0].num_fewshots, task_info[1].num_fewshots} == {0, 1}


def test_custom_tasks():
    """
    Tests that task info selector correctly handles custom tasks.
    """
    registry = Registry(tasks="custom|test_task_revision|0", custom_tasks="tests.unit.tasks.test_registry")

    assert registry.tasks_list == ["custom|test_task_revision|0"]
    assert set(registry.task_to_configs.keys()) == {"custom|test_task_revision"}

    task_info: list[LightevalTaskConfig] = registry.task_to_configs["custom|test_task_revision"]
    assert task_info[0].num_fewshots == 0


def test_superset_expansion():
    """
    Tests that task info selector correctly handles supersets.
    """
    registry = Registry(tasks="lighteval|storycloze|0")

    # The task list is saved as provided by the user
    assert registry.tasks_list == ["lighteval|storycloze|0"]

    # But we expand the superset when loading the configurations
    assert set(registry.task_to_configs.keys()) == {"lighteval|storycloze:2016", "lighteval|storycloze:2018"}

    for task_name in {"lighteval|storycloze:2016", "lighteval|storycloze:2018"}:
        task_info: list[LightevalTaskConfig] = registry.task_to_configs[task_name]
        assert task_info[0].num_fewshots == 0


def test_superset_with_subset_task():
    """
    Tests that task info selector correctly handles if both superset and one of subset tasks are provided.
    """
    registry = Registry(tasks="original|mmlu|3,original|mmlu:abstract_algebra|5")

    # We have all mmlu tasks
    assert set(registry.tasks_list) == {"original|mmlu|3", "original|mmlu:abstract_algebra|5"}
    assert len(registry.task_to_configs.keys()) == 57

    task_info: list[LightevalTaskConfig] = registry.task_to_configs["original|mmlu:abstract_algebra"]
    assert {task_info[0].num_fewshots, task_info[1].num_fewshots} == {3, 5}


def test_cli_sampling_params():
    """
    Tests task setting the sampling parameters in CLI.
    """
    registry_no_sampling = Registry(tasks="lighteval|math_500|0")

    task_info_no_sampling: list[LightevalTaskConfig] = registry_no_sampling.task_to_configs["lighteval|math_500"]
    # Default values
    assert task_info_no_sampling[0].metrics[0].sample_level_fn.k == 1
    assert task_info_no_sampling[0].metrics[0].sample_level_fn.n == 1

    registry = Registry(tasks="lighteval|math_500@k=2@n=10|0")

    task_info: list[LightevalTaskConfig] = registry.task_to_configs["lighteval|math_500"]
    assert task_info[0].metrics[0].sample_level_fn.k == 2
    assert task_info[0].metrics[0].sample_level_fn.n == 10


def test_cli_sampling_params_fail():
    """
    Tests task setting the sampling parameters in CLI failure when args are wrong.
    """
    # creation of object should fail
    with pytest.raises(ValueError):
        Registry("lighteval|math_500@plop|0")


def test_task_group_expansion_with_subset_expansion():
    """
    Tests that task info selector correctly handles a group with task superset is provided.
    """
    registry = Registry(tasks="all_mmlu", custom_tasks="tests.unit.tasks.test_registry")

    # We have all mmlu tasks
    assert len(registry.task_to_configs.keys()) == 57


def test_invalid_task_creation():
    """
    Tests that tasks info registry correctly raises errors for invalid tasks
    """
    with pytest.raises(ValueError):
        Registry(tasks="custom|task_revision")


def test_task_duplicates():
    """
    Tests that task info selector correctly handles if duplicate tasks are provided.
    """
    registry = Registry(
        tasks="custom|test_task_revision|0,custom|test_task_revision|0", custom_tasks="tests.unit.tasks.test_registry"
    )

    assert list(registry.tasks_list) == ["custom|test_task_revision|0"]


def test_task_creation():
    """
    Tests that tasks registry correctly creates tasks
    """
    registry = Registry(tasks="lighteval|storycloze:2016|0")
    task = registry.load_tasks()["lighteval|storycloze:2016|0"]

    assert isinstance(task, LightevalTask)
    assert task.name == "storycloze:2016"

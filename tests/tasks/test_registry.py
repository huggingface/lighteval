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
    "zero_and_one": "custom|test_task_revision|0|0,custom|test_task_revision|1|0",
    "all_mmlu": "original|mmlu|3|0",
}


def test_custom_task_groups():
    """
    Tests that task info selector correctly handles custom task groups.
    """
    registry = Registry(custom_tasks="tests.tasks.test_registry")
    task_info = registry.taskinfo_selector("zero_and_one")

    assert set(task_info.keys()) == {"custom|test_task_revision"}
    assert task_info["custom|test_task_revision"] == [
        {"fewshots": 0, "truncate_fewshots": False},
        {"fewshots": 1, "truncate_fewshots": False},
    ]


def test_custom_tasks():
    """
    Tests that task info selector correctly handles custom tasks.
    """
    registry = Registry(custom_tasks="tests.tasks.test_registry")
    task_info = registry.taskinfo_selector("custom|test_task_revision|0|0")

    assert list(task_info.keys()) == ["custom|test_task_revision"]
    assert task_info["custom|test_task_revision"] == [{"fewshots": 0, "truncate_fewshots": False}]


def test_superset_expansion():
    """
    Tests that task info selector correctly handles supersets.
    """
    registry = Registry()

    task_info = registry.taskinfo_selector("lighteval|storycloze|0|0")

    assert list(task_info.keys()) == ["lighteval|storycloze:2016", "lighteval|storycloze:2018"]
    assert task_info["lighteval|storycloze:2016"] == [{"fewshots": 0, "truncate_fewshots": False}] and task_info[
        "lighteval|storycloze:2018"
    ] == [{"fewshots": 0, "truncate_fewshots": False}]


def test_superset_with_subset_task():
    """
    Tests that task info selector correctly handles if both superset and one of subset tasks are provided.
    """
    registry = Registry()

    task_info = registry.taskinfo_selector("original|mmlu|3|0,original|mmlu:abstract_algebra|5|0")

    # We have all mmlu tasks
    assert len(task_info.keys()) == 57
    # Since it's defined twice
    assert task_info["original|mmlu:abstract_algebra"] == [
        {"fewshots": 3, "truncate_fewshots": False},
        {"fewshots": 5, "truncate_fewshots": False},
    ]


def test_task_group_expansion_with_subset_expansion():
    """
    Tests that task info selector correctly handles a group with task superset is provided.
    """
    registry = Registry(custom_tasks="tests.tasks.test_registry")

    task_info = registry.taskinfo_selector("all_mmlu")

    assert len(task_info.keys()) == 57


def test_invalid_task_creation():
    """
    Tests that tasks info registry correctly raises errors for invalid tasks
    """
    registry = Registry()
    with pytest.raises(ValueError):
        registry.get_tasks_configs("custom|task_revision")


def test_task_duplicates():
    """
    Tests that task info selector correctly handles if duplicate tasks are provided.
    """
    registry = Registry()

    task_info = registry.taskinfo_selector("custom|test_task_revision|0|0,custom|test_task_revision|0|0")

    assert list(task_info.keys()) == ["custom|test_task_revision"]


def test_task_creation():
    """
    Tests that tasks registry correctly creates tasks
    """
    registry = Registry()
    task_config = registry.get_tasks_configs("lighteval|storycloze:2016|0|0")
    task = registry.get_tasks_from_configs(task_config)["lighteval|storycloze:2016|0"]

    assert isinstance(task, LightevalTask)
    assert task.name == "storycloze:2016"

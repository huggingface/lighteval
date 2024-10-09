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
from lighteval.tasks.registry import Registry, taskinfo_selector


TASKS_TABLE = [
    LightevalTaskConfig(
        name="test_task_revision",
        # Won't be called, so it can be anything
        prompt_function=lambda x: x,  # type: ignore
        hf_repo="test",
        hf_subset="default",
        evaluation_splits=["train"],
        metric=[],
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
    tasks, task_info = taskinfo_selector("zero_and_one", registry)

    assert set(tasks) == {"custom|test_task_revision"}
    assert all(task in task_info for task in tasks)
    assert all(task_info[task] == [(1, False), (0, False)] for task in tasks)


def test_custom_tasks():
    """
    Tests that task info selector correctly handles custom tasks.
    """
    registry = Registry(custom_tasks="tests.tasks.test_registry")
    tasks, task_info = taskinfo_selector("custom|test_task_revision|0|0", registry)

    assert tasks == ["custom|test_task_revision"]
    assert task_info["custom|test_task_revision"] == [(0, False)]


def test_superset_expansion():
    """
    Tests that task info selector correctly handles supersets.
    """
    registry = Registry()

    tasks, task_info = taskinfo_selector("lighteval|storycloze|0|0", registry)

    assert set(tasks) == {"lighteval|storycloze:2016", "lighteval|storycloze:2018"}
    assert all(task_info[task] == [(0, False)] for task in tasks)


def test_superset_with_subset_task():
    """
    Tests that task info selector correctly handles if both superset and one of subset tasks are provided.
    """
    registry = Registry()

    tasks, task_info = taskinfo_selector("original|mmlu|3|0,original|mmlu:abstract_algebra|5|0", registry)

    # We have all mmlu tasks
    assert len(tasks) == 57
    # Since it's defined twice
    assert task_info["original|mmlu:abstract_algebra"] == [(5, False), (3, False)]


def test_task_group_expansion_with_subset_expansion():
    """
    Tests that task info selector correctly handles a group with task superset is provided.
    """
    registry = Registry(custom_tasks="tests.tasks.test_registry")

    tasks = taskinfo_selector("all_mmlu", registry)[0]

    assert len(tasks) == 57


def test_invalid_task_creation():
    """
    Tests that tasks info registry correctly raises errors for invalid tasks
    """
    registry = Registry()
    with pytest.raises(ValueError):
        registry.get_task_dict(["custom|task_revision"])


def test_task_duplicates():
    """
    Tests that task info selector correctly handles if duplicate tasks are provided.
    """
    registry = Registry()

    tasks, task_info = taskinfo_selector("custom|test_task_revision|0|0,custom|test_task_revision|0|0", registry)

    assert tasks == ["custom|test_task_revision"]
    assert task_info["custom|test_task_revision"] == [(0, False)]


def test_task_creation():
    """
    Tests that tasks registry correctly creates tasks
    """
    registry = Registry()
    task_info = registry.get_task_dict(["lighteval|storycloze:2016"])["lighteval|storycloze:2016"]

    assert isinstance(task_info, LightevalTask)
    assert task_info.name == "lighteval|storycloze:2016"

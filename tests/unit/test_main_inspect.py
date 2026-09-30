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

from unittest.mock import MagicMock, patch

from lighteval.main_inspect import eval


# Shared with slow tests: a real custom tasks module that defines gsm8k_test.
CUSTOM_TASKS_PATH = "examples/custom_tasks_tests.py"


def test_inspect_eval_forwards_custom_tasks_to_registry():
    """Inspect eval must load tasks from the custom_tasks argument.

    Without forwarding custom_tasks into Registry, a task that exists only in
    the custom module cannot be resolved.
    """
    seen_configs = []

    def capture_get_inspect_ai_task(task_config, epochs=1, epochs_reducer=None):
        seen_configs.append(task_config)
        return MagicMock(name=f"inspect-task-{task_config.name}")

    with (
        patch("lighteval.main_inspect.get_inspect_ai_task", side_effect=capture_get_inspect_ai_task),
        patch("lighteval.main_inspect.inspect_ai_eval_set", return_value=(True, [])),
    ):
        eval(
            models=["mock-model"],
            tasks="gsm8k_test|0",
            custom_tasks=CUSTOM_TASKS_PATH,
            log_dir="unused-log-dir",
        )

    assert any(config.name == "gsm8k_test" for config in seen_configs), (
        "Registry did not load the custom task; custom_tasks was likely ignored"
    )

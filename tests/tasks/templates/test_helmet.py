# MIT License

# Copyright (c) 2025 The HuggingFace Team

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

from lighteval.tasks.templates.helmet import HelmetTask


@pytest.fixture
def helmet_task():
    return HelmetTask()


def test_prompts_loaded(helmet_task):
    """
    Tests that json files stored in helmet_data are loaded correctly.
    """
    assert len(helmet_task.prompts) > 0, "No prompts were loaded"
    for fname, data in helmet_task.prompts.items():
        assert isinstance(data, dict), f"{fname} did not load as a dict"
        assert "instruction" in data or "demos" in data, f"{fname} missing required keys"


def test_get_prompt(helmet_task):
    """
    Tests that get_prompt returns the correct dictionary.
    """
    for fname in helmet_task.prompts.keys():
        prompt = helmet_task.get_prompt(fname)
        assert prompt == helmet_task.prompts[fname], f"get_prompt failed for {fname}"

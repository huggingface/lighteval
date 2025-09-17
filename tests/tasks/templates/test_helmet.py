import os
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
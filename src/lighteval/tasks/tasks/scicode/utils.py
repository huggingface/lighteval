"""Utility functions for SciCode.

Based on original implementation:
https://github.com/scicode-bench/SciCode
"""

import re
from pathlib import Path


def extract_python_script(response: str) -> str:
    """Extract Python code from markdown code blocks."""
    if "```" in response:
        if "```python" in response:
            python_script = response.split("```python")[1].split("```")[0]
        else:
            python_script = response.split("```")[1].split("```")[0]
    else:
        python_script = response

    python_script = re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", python_script, flags=re.MULTILINE)
    return python_script


def get_h5py_file_path() -> Path:
    """Get path to test_data.h5, downloading from HuggingFace if necessary.

    Note: Currently hosted at akshathmangudi/scicode-files.
    Once official hosting is available, this will be updated to the official repository.
    """
    from huggingface_hub import hf_hub_download

    repo_id = "akshathmangudi/scicode-files"
    try:
        h5py_file = hf_hub_download(
            repo_id=repo_id,
            filename="test_data.h5",
            repo_type="dataset",
        )
        return Path(h5py_file)
    except Exception as e:
        # Fallback: check local path
        local_path = Path(__file__).parent / "test_data.h5"
        if local_path.exists():
            return local_path
        raise FileNotFoundError(
            f"Could not download test_data.h5 from {repo_id}. Please ensure it's available or place it at {local_path}"
        ) from e

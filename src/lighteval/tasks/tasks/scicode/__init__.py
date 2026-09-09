"""SciCode benchmark implementation for Lighteval.

Based on the original SciCode implementation:
https://github.com/scicode-bench/SciCode/blob/main/eval/inspect_ai/scicode.py
"""

from lighteval.tasks.tasks.scicode.main import TASKS_TABLE, scicode


__all__ = ["scicode", "TASKS_TABLE"]

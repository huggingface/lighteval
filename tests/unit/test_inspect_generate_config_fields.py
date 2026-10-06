# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""`lighteval eval` forwards generation options to `inspect_ai.eval_set`, which
passes any argument it does not itself declare on to `GenerateConfig`.

A name that `GenerateConfig` does not accept therefore fails at run time with a
pydantic ValidationError rather than at import time, and only once a user
actually runs an eval. Issue #1327 was exactly that: `frequence_penalty`,
`log_probs` and `response_format` stopped being valid, so a fresh install of
lighteval could not run the Inspect-backed eval path.

This test reads the keyword names straight out of the `inspect_ai_eval_set(...)`
call in `main_inspect.py` rather than repeating them here, so it keeps testing
the real call site as that call site changes.
"""

import ast
import inspect
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig

import lighteval.main_inspect as main_inspect


CALL_NAME = "inspect_ai_eval_set"


def _forwarded_keywords() -> set[str]:
    """Keyword names passed to `inspect_ai_eval_set` in `main_inspect.py`."""
    source = Path(inspect.getfile(main_inspect)).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == CALL_NAME:
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"no call to {CALL_NAME}() found in main_inspect.py")


def test_forwarded_options_are_accepted_downstream():
    forwarded = _forwarded_keywords()
    assert forwarded, "expected keyword arguments on the eval_set call"

    declared = set(inspect.signature(eval_set).parameters)
    passed_through = forwarded - declared

    unknown = passed_through - set(GenerateConfig.model_fields)
    assert not unknown, (
        f"{CALL_NAME}() is called with {sorted(unknown)}, which inspect_ai.eval_set "
        f"does not declare and GenerateConfig does not accept. With the installed "
        f"inspect-ai these reach GenerateConfig and raise a ValidationError at run "
        f"time. Rename them to the current GenerateConfig fields, or stop passing them."
    )

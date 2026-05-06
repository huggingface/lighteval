# MIT License
# Copyright (c) 2024 The HuggingFace Team

"""
Minimal loader for the TVD-MI paired-response synthetic example.

This module intentionally avoids tight coupling to task registries so it can be
used as a simple reference/template. It provides `read_jsonl()` and `build_docs()`
helpers to construct lighteval `Doc` objects with the fields expected by TVD-MI.

Expected JSONL schema per line:
  - response_a: str
  - response_b: str
  - pair_label: int (1=same, 0=different)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


try:
    # lighteval Doc type (preferred if available)
    from lighteval.tasks.requests import Doc  # type: ignore
except Exception:
    # Fallback: minimal doc type for local testing / documentation purposes
    @dataclass
    class Doc:  # type: ignore
        query: str = ""
        choices: list[str] | None = None
        gold_index: int | list[int] | None = None
        task_name: str | None = None
        specific: dict[str, Any] | None = None


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = HERE / "tvd_mi_synthetic.jsonl"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e
    return rows


def _set_attr_if_possible(obj: Any, name: str, value: Any) -> None:
    """
    Try to set `obj.name = value`. Some Doc implementations may forbid new attributes.
    """
    try:
        setattr(obj, name, value)
    except Exception:
        # It's fine if Doc is strict; we always store in `specific` too.
        pass


def build_docs(rows: Iterable[dict[str, Any]], task_name: str = "tvd_mi_synthetic") -> list[Doc]:
    docs: list[Doc] = []
    for i, r in enumerate(rows):
        if "response_a" not in r or "response_b" not in r or "pair_label" not in r:
            raise ValueError(
                f"Row {i} missing required keys. Expected response_a/response_b/pair_label. Got keys={list(r.keys())}"
            )

        response_a = str(r["response_a"])
        response_b = str(r["response_b"])
        pair_label = int(r["pair_label"])

        # Create a minimal Doc. Many metrics/tests assume `query`/`choices` exist.
        doc = Doc(
            query="",
            choices=[],
            gold_index=0,
            task_name=task_name,
            specific={
                "response_a": response_a,
                "response_b": response_b,
                "pair_label": pair_label,
            },
        )

        # Also set direct attributes for compatibility with JudgeLLMTVDMI.compute as currently implemented.
        _set_attr_if_possible(doc, "response_a", response_a)
        _set_attr_if_possible(doc, "response_b", response_b)
        _set_attr_if_possible(doc, "pair_label", pair_label)

        docs.append(doc)

    return docs


def load_default_docs() -> list[Doc]:
    """
    Convenience helper to load the default JSONL shipped with this example folder.
    """
    rows = read_jsonl(DEFAULT_DATA_PATH)
    return build_docs(rows)


if __name__ == "__main__":
    docs = load_default_docs()
    print(f"Loaded {len(docs)} docs from {DEFAULT_DATA_PATH}")
    print(
        "First doc has attrs:",
        hasattr(docs[0], "response_a"),
        hasattr(docs[0], "response_b"),
        hasattr(docs[0], "pair_label"),
    )
    print("First doc specific keys:", list((docs[0].specific or {}).keys()))

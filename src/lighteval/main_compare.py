# MIT License
# Copyright (c) 2026 The HuggingFace Team

"""Offline comparison command for saved evaluation runs."""

import json
from pathlib import Path

import typer
from typing_extensions import Annotated

from lighteval.comparison import compare_runs, load_run


def compare(
    baseline: Annotated[
        Path, typer.Argument(exists=True, help="Baseline Inspect log or Lighteval details file/directory.")
    ],
    candidate: Annotated[Path, typer.Argument(exists=True, help="Candidate log or details file/directory.")],
    metric: Annotated[list[str], typer.Option(help="Scalar mean metric to compare. Repeat for multiple metrics.")],
    lower_is_better: Annotated[
        list[str] | None, typer.Option(help="Selected metrics where smaller is better.")
    ] = None,
    confidence_level: Annotated[float, typer.Option(min=0, max=1)] = 0.95,
    n_resamples: Annotated[int, typer.Option(min=2)] = 9999,
    seed: Annotated[int, typer.Option(help="Seed for deterministic resampling.")] = 0,
    allow_partial: Annotated[
        bool, typer.Option(help="Compare only valid shared observations; report exclusions.")
    ] = False,
    fail_on_regression: Annotated[
        bool, typer.Option(help="Exit with code 1 when a Holm-adjusted regression is detected.")
    ] = False,
    output: Annotated[
        Path | None, typer.Option(help="Save the complete JSON report, including changed sample outputs.")
    ] = None,
):
    """Compare paired sample scores, uncertainty, and changed examples without running a model."""
    if output is not None and any(
        output.resolve() == path.resolve() or (path.is_dir() and output.resolve().is_relative_to(path.resolve()))
        for path in (baseline, candidate)
    ):
        raise typer.BadParameter(
            "Save the report outside the input files/directories so it cannot overwrite or pollute a run"
        )
    try:
        report = compare_runs(
            load_run(baseline),
            load_run(candidate),
            metric,
            lower_is_better=lower_is_better or (),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed,
            allow_partial=allow_partial,
        )
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
    typer.echo(report.to_markdown())
    if fail_on_regression and any(row.verdict == "regressed" for row in report.comparisons):
        raise typer.Exit(code=1)

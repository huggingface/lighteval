# MIT License
# Copyright (c) 2026 The HuggingFace Team

"""Paired comparisons of saved, sample-level evaluation scores.

No task registry, model, dataset download, or scoring API is used. Only arithmetic
means of scalar sample scores are supported; corpus metrics need their own paired
resampling procedure and must not be reconstructed by averaging contributions.
"""

import hashlib
import importlib.metadata
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel
from scipy.stats import binomtest, bootstrap, permutation_test


@dataclass(frozen=True)
class ScoredSample:
    """One saved observation. Epochs of the same sample form one statistical unit."""

    task: str
    sample_id: str | int
    scores: dict[str, Any]
    input: Any
    target: Any
    epoch: int = 1
    output: Any = None
    error: bool = False

    @property
    def key(self) -> tuple[str, str, int]:
        """Use typed IDs rather than position or a lossy string conversion."""
        if isinstance(self.sample_id, bool) or not isinstance(self.sample_id, (str, int)) or self.sample_id == "":
            raise ValueError(f"Task {self.task!r} has a missing or invalid sample ID: {self.sample_id!r}")
        return self.task, json.dumps(self.sample_id), self.epoch


@dataclass
class EvaluationRun:
    """Samples and provenance of one evaluation run."""

    source: str
    samples: list[ScoredSample]
    format: str = "python"
    status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskComparison:
    """A paired comparison for one task and scalar metric."""

    task: str
    metric: str
    higher_is_better: bool
    samples: int
    observations: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    confidence_interval: tuple[float, float] | None
    p_value: float
    test: str
    improved: int
    regressed: int
    unchanged: int
    baseline_only: int
    candidate_only: int
    excluded: int
    adjusted_p_value: float = 1.0
    verdict: str = "inconclusive"
    changes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Serializable report; p-values are adjusted across every returned row."""

    baseline: str
    candidate: str
    confidence_level: float
    n_resamples: int
    seed: int
    allow_partial: bool
    comparisons: list[TaskComparison]
    baseline_only_tasks: list[str]
    candidate_only_tasks: list[str]
    baseline_metadata: dict[str, Any] = field(default_factory=dict)
    candidate_metadata: dict[str, Any] = field(default_factory=dict)
    unselected_tasks: list[str] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict:
        """Return a JSON-compatible report with complete sample-change details."""
        return _json_value(asdict(self))

    def to_markdown(self) -> str:
        """Render a compact report without rounding raw JSON results."""

        def escape(value):
            return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

        lines = [
            "# Paired evaluation comparison",
            "",
            f"Baseline: {escape(self.baseline)}",
            f"Candidate: {escape(self.candidate)}",
            "",
            "Delta is candidate minus baseline. Repeated epochs are averaged within each sample before inference.",
            "P-values use Holm correction across all task/metric comparisons.",
            "",
            f"| Task | Metric | Samples | Baseline | Candidate | Delta | {self.confidence_level:.0%} paired CI | Adjusted p | Verdict |",
            "|---|---|---:|---:|---:|---:|---|---:|---|",
        ]
        for row in self.comparisons:
            ci = (
                "unavailable (one sample)"
                if row.confidence_interval is None
                else (f"[{row.confidence_interval[0]:.4f}, {row.confidence_interval[1]:.4f}]")
            )
            lines.append(
                f"| {escape(row.task)} | {escape(row.metric)} | {row.samples} | {row.baseline_mean:.4f} | "
                f"{row.candidate_mean:.4f} | {row.delta:+.4f} | {ci} | {row.adjusted_p_value:.4g} | {row.verdict} |"
            )
        lines.extend(["", "## Pairing and coverage", ""])
        for row in self.comparisons:
            lines.append(
                f"- {escape(row.task)} / {escape(row.metric)}: {row.observations} paired observations, "
                f"{row.improved} improved, {row.regressed} regressed, {row.unchanged} unchanged samples; "
                f"{row.baseline_only} baseline-only, {row.candidate_only} candidate-only, {row.excluded} invalid pairs."
            )
        if self.baseline_only_tasks or self.candidate_only_tasks:
            lines.append(f"- Baseline-only tasks: {escape(self.baseline_only_tasks)}")
            lines.append(f"- Candidate-only tasks: {escape(self.candidate_only_tasks)}")
        if self.unselected_tasks:
            lines.append(f"- Tasks without a selected metric: {escape(self.unselected_tasks)}")
        lines.extend(
            [
                "",
                "Intervals are paired percentile bootstrap intervals for mean differences, not simultaneous intervals. "
                "They describe uncertainty over the evaluated sample population; they do not capture training-seed, "
                "dataset contamination, judge, or prompt-selection uncertainty. Small or constant samples can produce "
                "degenerate intervals. Only scalar mean metrics are supported.",
                "",
            ]
        )
        return "\n".join(lines)


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _input_fingerprint(value: Any) -> str:
    value = _json_value(value)
    if isinstance(value, list):
        # Inspect assigns a fresh message ID when the same dataset is loaded for
        # another run. These bookkeeping fields are not part of the model input.
        value = [
            {key: item for key, item in message.items() if key not in {"id", "source", "metadata"}}
            if isinstance(message, dict) and "role" in message
            else message
            for message in value
        ]
    return _fingerprint(value)


def _numeric(value: Any) -> float | None:
    if isinstance(value, str):
        if value in {"C", "I", "P", "N"}:
            value = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}[value]
        else:
            return None
    if not isinstance(value, (Real, np.bool_)):
        return None
    return float(value) if math.isfinite(value) else None


def _load_details(file: Path) -> EvaluationRun:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(file)
    required = {"doc", "model_response", "metric"}
    if not required.issubset(table.schema_arrow.names):
        raise ValueError(f"{file} is not a Lighteval details file (expected doc/model_response/metric columns)")
    # The filename carries the few-shot configuration, unlike doc.task_name.
    name = re.fullmatch(r"details_(.+)_\d{4}-\d{2}-\d{2}T.+", file.stem)
    task = name.group(1) if name else None
    samples = []
    for batch in table.iter_batches(columns=sorted(required)):
        for row in batch.to_pylist():
            doc = row["doc"]
            indices = doc["gold_index"]
            indices = indices if isinstance(indices, list) else [indices]
            target = [doc["choices"][i] for i in indices] if doc["choices"] else None
            samples.append(
                ScoredSample(
                    task=task or doc["task_name"],
                    sample_id=doc.get("id", ""),
                    scores=row["metric"],
                    input=doc["query"],
                    target=target,
                    output=row["model_response"].get("text"),
                )
            )
    return EvaluationRun(str(file), samples, "lighteval-parquet")


def _load_inspect(file: Path) -> EvaluationRun:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(file))
    if not log.samples:
        raise ValueError(f"{file} has no saved sample scores; header-only logs cannot be compared")
    samples = []
    for sample in log.samples:
        scores = {}
        for name, score in (sample.scores or {}).items():
            if isinstance(score.value, dict):
                scores.update({f"{name}.{key}": value for key, value in score.value.items()})
            else:
                scores[name] = score.value
        samples.append(
            ScoredSample(
                task=log.eval.task,
                sample_id=sample.id,
                epoch=sample.epoch,
                scores=scores,
                input=_json_value(sample.input),
                target=sample.target,
                output=sample.output.completion if sample.output else None,
                error=sample.error is not None,
            )
        )
    return EvaluationRun(
        str(file),
        samples,
        "inspect",
        log.status,
        {
            "model": log.eval.model,
            "task": log.eval.task,
            "task_version": log.eval.task_version,
        },
    )


def _file_digest(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: str | Path) -> EvaluationRun:
    """Read one Inspect .eval/.json log or Lighteval details Parquet file.

    Directories may contain one run's files, never multiple timestamps/models of
    the same task. Duplicate (task, ID, epoch) observations are rejected at pairing.
    """
    path = Path(path)
    files = (
        sorted(p for p in path.rglob("*") if p.suffix in {".eval", ".json", ".parquet"}) if path.is_dir() else [path]
    )
    if not files:
        raise ValueError(f"No Inspect logs or Lighteval detail files found at {path}")
    runs = []
    for file in files:
        if file.suffix == ".parquet":
            runs.append(_load_details(file))
        elif file.suffix in {".eval", ".json"}:
            runs.append(_load_inspect(file))
        else:
            raise ValueError(f"Unsupported run file {file}; expected .eval, .json, or .parquet")
    models = {run.metadata["model"] for run in runs if run.format == "inspect"}
    if len(models) > 1:
        raise ValueError("Input directory contains Inspect logs from different models; select one model's run")
    return EvaluationRun(
        str(path),
        [sample for run in runs for sample in run.samples],
        "+".join(sorted({run.format for run in runs})),
        "success" if all(run.status == "success" for run in runs) else "incomplete",
        {
            "files": [
                {"path": str(file), "sha256": _file_digest(file), **run.metadata} for file, run in zip(files, runs)
            ],
            "versions": {name: importlib.metadata.version(name) for name in ("lighteval", "numpy", "scipy")},
        },
    )


def _index(run: EvaluationRun) -> dict:
    index = {}
    for sample in run.samples:
        if not sample.task:
            raise ValueError(f"{run.source}: every observation must have a task name")
        if sample.key in index:
            raise ValueError(f"{run.source}: duplicate task/sample/epoch {sample.key}. Select a single run.")
        index[sample.key] = sample
    if not index:
        raise ValueError(f"{run.source}: no samples to compare")
    return index


def _paired_statistics(baseline, candidate, confidence_level, n_resamples, seed):
    differences = candidate - baseline
    if len(differences) == 1:
        interval = None
    elif np.all(differences == differences[0]):
        interval = (float(differences[0]), float(differences[0]))
    else:
        # Bounded temporary memory even for large evaluations; paired differences
        # are resampled as a single array, so the pairing cannot be broken.
        batch = max(1, min(128, 1_000_000 // len(differences)))
        result = bootstrap(
            (differences,),
            np.mean,
            vectorized=True,
            method="percentile",
            batch=batch,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=np.random.default_rng(seed),
        )
        interval = (float(result.confidence_interval.low), float(result.confidence_interval.high))

    binary = np.isin(baseline, [0, 1]).all() and np.isin(candidate, [0, 1]).all()
    if binary:
        wins = int(np.sum(differences > 0))
        losses = int(np.sum(differences < 0))
        p_value = float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0
        test = "exact_mcnemar"
    elif np.all(differences == 0) or len(differences) == 1:
        p_value, test = 1.0, "paired_permutation"
    else:
        result = permutation_test(
            (differences,),
            np.mean,
            permutation_type="samples",
            vectorized=True,
            n_resamples=n_resamples,
            batch=max(1, min(128, 1_000_000 // len(differences))),
            random_state=np.random.default_rng(seed),
        )
        p_value, test = float(result.pvalue), "paired_permutation"
    return interval, p_value, test


def _compare_metric(
    task,
    metric,
    common,
    left,
    right,
    *,
    lower_is_better,
    confidence_level,
    n_resamples,
    seed,
    allow_partial,
    baseline_only,
    candidate_only,
):
    groups = defaultdict(list)
    excluded = 0
    for key in common:
        a, b = left[key], right[key]
        a_score, b_score = _numeric(a.scores.get(metric)), _numeric(b.scores.get(metric))
        if a.error or b.error or a_score is None or b_score is None:
            if not allow_partial:
                raise ValueError(
                    f"{task}: invalid or missing metric {metric!r} at sample/epoch {key[1:]}; use allow_partial=True to explicitly exclude it"
                )
            excluded += 1
            continue
        groups[key[1]].append((a, b, a_score, b_score))
    if not groups:
        raise ValueError(f"{task}: no valid paired observations for metric {metric!r}")
    values = []
    changes = []
    for sample_id, observations in sorted(groups.items()):
        a_score = float(np.mean([obs[2] for obs in observations]))
        b_score = float(np.mean([obs[3] for obs in observations]))
        values.append((a_score, b_score))
        if a_score != b_score:
            changes.append(
                {
                    "sample_id": observations[0][0].sample_id,
                    "baseline": a_score,
                    "candidate": b_score,
                    "delta": b_score - a_score,
                    "epochs": [obs[0].epoch for obs in observations],
                    "input": observations[0][0].input,
                    "target": observations[0][0].target,
                    "baseline_outputs": [obs[0].output for obs in observations],
                    "candidate_outputs": [obs[1].output for obs in observations],
                }
            )
    a_values, b_values = np.array(values).T
    delta_values = b_values - a_values
    row_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{task}\0{metric}".encode()).digest()[:8], "big")
    interval, p_value, test = _paired_statistics(a_values, b_values, confidence_level, n_resamples, row_seed)
    direction = -1 if lower_is_better else 1
    return TaskComparison(
        task=task,
        metric=metric,
        higher_is_better=direction == 1,
        samples=len(groups),
        observations=sum(len(obs) for obs in groups.values()),
        baseline_mean=float(a_values.mean()),
        candidate_mean=float(b_values.mean()),
        delta=float(delta_values.mean()),
        confidence_interval=interval,
        p_value=p_value,
        test=test,
        improved=int(np.sum(direction * delta_values > 0)),
        regressed=int(np.sum(direction * delta_values < 0)),
        unchanged=int(np.sum(delta_values == 0)),
        baseline_only=baseline_only,
        candidate_only=candidate_only,
        excluded=excluded,
        changes=changes,
    )


def _align_task(task, left, right, allow_partial):
    left_keys = {key for key in left if key[0] == task}
    right_keys = {key for key in right if key[0] == task}
    if left_keys != right_keys and not allow_partial:
        raise ValueError(
            f"{task}: sample/epoch sets differ ({len(left_keys - right_keys)} baseline-only, {len(right_keys - left_keys)} candidate-only)"
        )
    common = sorted(left_keys & right_keys)
    for key in common:
        if _input_fingerprint(left[key].input) != _input_fingerprint(right[key].input) or _fingerprint(
            left[key].target
        ) != _fingerprint(right[key].target):
            raise ValueError(
                f"{task}: input or target differs for sample/epoch {key[1:]}; IDs alone are insufficient for pairing"
            )
    return common, len(left_keys - right_keys), len(right_keys - left_keys)


def _adjust_holm(comparisons, confidence_level):
    # Holm step-down adjusted p-values must be monotone in sorted raw p-values.
    running_max = 0.0
    for rank, row in enumerate(sorted(comparisons, key=lambda item: item.p_value)):
        running_max = min(1.0, max(running_max, (len(comparisons) - rank) * row.p_value))
        row.adjusted_p_value = running_max
        if running_max <= 1 - confidence_level:
            oriented_delta = row.delta if row.higher_is_better else -row.delta
            row.verdict = "improved" if oriented_delta > 0 else "regressed" if oriented_delta < 0 else "inconclusive"


def _select_metrics(left, right, metrics):
    available = defaultdict(set)
    for sample in (*left.values(), *right.values()):
        available[sample.task].update(sample.scores)
    all_metrics = set().union(*available.values())
    missing = set(metrics) - all_metrics
    if missing:
        raise ValueError(
            f"Requested invalid or missing metric(s) {sorted(missing)}; available metrics: {sorted(all_metrics)}"
        )
    return {task: [metric for metric in metrics if metric in names] for task, names in available.items()}


def _validate_options(metrics, lower_is_better, confidence_level, n_resamples):
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Select at least one metric, without duplicates")
    if not set(lower_is_better).issubset(metrics):
        raise ValueError("lower_is_better must refer to selected metrics")
    if not 0 < confidence_level < 1 or n_resamples < 2:
        raise ValueError("confidence_level must be between 0 and 1 and n_resamples must be at least 2")


def compare_runs(
    baseline: EvaluationRun,
    candidate: EvaluationRun,
    metrics: Sequence[str],
    *,
    lower_is_better: Sequence[str] = (),
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 0,
    allow_partial: bool = False,
) -> ComparisonReport:
    """Compare arithmetic means on strictly paired sample IDs and epochs.

    By default, require the same tasks/IDs/epochs, successful runs, matching
    questions and targets, and finite scores on both sides. ``allow_partial``
    explicitly selects the intersection and reports every excluded observation.
    Repeated epochs are averaged per document to avoid treating correlated
    generations as independent examples. Holm correction controls the familywise
    error rate across returned task/metric rows; intervals remain marginal.
    """
    _validate_options(metrics, lower_is_better, confidence_level, n_resamples)
    if not allow_partial and (baseline.status != "success" or candidate.status != "success"):
        raise ValueError("Cannot compare incomplete runs without allow_partial=True")
    left, right = _index(baseline), _index(candidate)
    selected_metrics = _select_metrics(left, right, metrics)
    left_tasks = {key[0] for key in left}
    right_tasks = {key[0] for key in right}
    if left_tasks != right_tasks and not allow_partial:
        raise ValueError(
            f"Task sets differ: baseline-only {sorted(left_tasks - right_tasks)}, candidate-only {sorted(right_tasks - left_tasks)}"
        )
    comparisons = []
    for task in sorted(left_tasks & right_tasks):
        if not selected_metrics[task]:
            continue
        common, baseline_only, candidate_only = _align_task(task, left, right, allow_partial)
        for metric in selected_metrics[task]:
            comparisons.append(
                _compare_metric(
                    task,
                    metric,
                    common,
                    left,
                    right,
                    lower_is_better=metric in lower_is_better,
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    seed=seed,
                    allow_partial=allow_partial,
                    baseline_only=baseline_only,
                    candidate_only=candidate_only,
                )
            )
    if not comparisons:
        raise ValueError("No common tasks with selected metrics to compare")
    _adjust_holm(comparisons, confidence_level)
    return ComparisonReport(
        baseline.source,
        candidate.source,
        confidence_level,
        n_resamples,
        seed,
        allow_partial,
        comparisons,
        sorted(left_tasks - right_tasks),
        sorted(right_tasks - left_tasks),
        {**baseline.metadata, "format": baseline.format, "status": baseline.status},
        {**candidate.metadata, "format": candidate.format, "status": candidate.status},
        unselected_tasks=sorted(task for task in left_tasks & right_tasks if not selected_metrics[task]),
    )

# MIT License
# Copyright (c) 2026 The HuggingFace Team

"""Behavioral, statistical, and real-file tests for paired run comparison."""

import asyncio
import json
from dataclasses import asdict, replace
from itertools import product

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText, ModelOutput, get_model
from inspect_ai.scorer import exact
from inspect_ai.solver import generate
from typer.testing import CliRunner

from lighteval.comparison import EvaluationRun, ScoredSample, compare_runs, load_run
from lighteval.logging.info_loggers import DetailsLogger
from lighteval.main_compare import compare
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def run(scores, *, task="task", epochs=1):
    return EvaluationRun(
        "test",
        [
            ScoredSample(task, i, {"accuracy": score}, f"question {i}", "gold", epoch=epoch)
            for i, score in enumerate(scores)
            for epoch in range(1, epochs + 1)
        ],
    )


def row(a, b, **kwargs):
    return compare_runs(a, b, ["accuracy"], n_resamples=499, **kwargs).comparisons[0]


def test_exact_mcnemar_known_discordant_table():
    result = row(run([0] * 10), run([1] * 10))
    assert result.samples == result.observations == 10
    assert result.delta == 1
    assert result.confidence_interval == (1, 1)
    assert result.p_value == pytest.approx(2 / 2**10)
    assert result.adjusted_p_value == result.p_value
    assert result.test == "exact_mcnemar"
    assert result.verdict == "improved"
    assert (result.improved, result.regressed, result.unchanged) == (10, 0, 0)


def test_tied_binary_runs_have_no_evidence_of_change():
    result = row(run([0, 1, 0, 1]), run([1, 0, 0, 1]))
    assert result.delta == 0
    assert result.p_value == 1
    assert result.verdict == "inconclusive"


def test_identical_runs_are_exactly_zero():
    a = run([0.2, 0.4, 0.9, 1.2])
    result = row(a, a)
    assert result.delta == 0
    assert result.confidence_interval == (0, 0)
    assert result.p_value == 1
    assert not result.changes


def test_continuous_paired_permutation_exact_small_sample():
    result = row(run([0.0] * 5), run([0.5] * 5))
    assert result.test == "paired_permutation"
    assert result.p_value == pytest.approx(2 / 2**5)
    assert result.confidence_interval == (0.5, 0.5)


def test_bootstrap_preserves_pairing_and_matches_swap_symmetry():
    a = run([0.1, 4.2, 9.3, 16.4, 30.5, 50.6])
    b = run([0.2, 4.4, 9.6, 16.8, 31.0, 51.2])
    forward = row(a, b)
    backward = row(b, a)
    assert forward.delta == pytest.approx(-backward.delta)
    assert forward.p_value == backward.p_value
    assert forward.confidence_interval[0] == pytest.approx(-backward.confidence_interval[1])
    assert forward.confidence_interval[1] == pytest.approx(-backward.confidence_interval[0])
    assert forward.confidence_interval[1] - forward.confidence_interval[0] < 0.6


def test_reordering_samples_does_not_change_statistics():
    a, b = run([0.0, 0.2, 0.3, 0.6, 0.9]), run([0.2, 0.5, 0.2, 0.8, 0.8])
    expected = asdict(row(a, b))
    a.samples.reverse()
    b.samples = b.samples[2:] + b.samples[:2]
    assert asdict(row(a, b)) == expected


def test_repeated_epochs_are_not_independent_samples():
    single = row(run([0] * 5), run([1] * 5))
    repeated = row(run([0] * 5, epochs=20), run([1] * 5, epochs=20))
    assert repeated.samples == 5
    assert repeated.observations == 100
    assert repeated.p_value == single.p_value == pytest.approx(0.0625)
    assert repeated.verdict == "inconclusive"


def test_epochs_are_averaged_before_inference():
    a = run([0, 0, 0], epochs=2)
    b = replace(a, samples=[replace(s, scores={"accuracy": s.epoch - 1}) for s in a.samples])
    result = row(a, b)
    assert result.candidate_mean == 0.5
    assert result.samples == 3
    assert result.test == "paired_permutation"
    assert result.p_value == pytest.approx(0.25)


def test_holm_correction_covers_all_requested_tasks_and_metrics():
    a = run([0] * 6, task="a")
    b = run([1] * 6, task="a")
    a.samples += run([0] * 5, task="b").samples
    b.samples += run([1] * 5, task="b").samples
    results = compare_runs(a, b, ["accuracy"], n_resamples=99).comparisons
    assert results[0].p_value == pytest.approx(0.03125)
    assert [r.adjusted_p_value for r in results] == pytest.approx([0.0625, 0.0625])
    assert all(r.verdict == "inconclusive" for r in results)


def test_metrics_can_select_different_tasks_and_report_unselected_tasks():
    a, b = run([0] * 10), run([1] * 10)
    a.samples += [replace(s, scores={"loss": 1.5}) for s in run([0] * 10, task="loss_task").samples]
    b.samples += [replace(s, scores={"loss": 1.0}) for s in run([0] * 10, task="loss_task").samples]
    report = compare_runs(a, b, ["accuracy", "loss"], lower_is_better=["loss"], n_resamples=99)
    assert len(report.comparisons) == 2
    assert all(r.verdict == "improved" for r in report.comparisons)
    selected = compare_runs(a, b, ["accuracy"], n_resamples=99)
    assert len(selected.comparisons) == 1
    assert selected.unselected_tasks == ["loss_task"]


def test_lower_is_better_reverses_verdict_but_not_reported_delta():
    result = row(run([1] * 10), run([0] * 10), lower_is_better=["accuracy"])
    assert result.delta == -1
    assert result.verdict == "improved"
    assert result.improved == 10


def test_single_sample_has_no_confidence_interval():
    result = row(run([0.1]), run([0.4]))
    assert result.confidence_interval is None
    assert result.p_value == 1


def test_numpy_scalar_scores_work_in_the_python_api():
    result = row(run([np.float32(0.5), np.int64(0)]), run([np.float32(0.8), np.bool_(True)]))
    assert result.candidate_mean == pytest.approx(0.9)


@pytest.mark.parametrize("field", ["input", "target"])
def test_matching_ids_cannot_hide_different_examples(field):
    a = run([0, 1])
    b = replace(a, samples=[replace(a.samples[0], **{field: "changed"}), a.samples[1]])
    with pytest.raises(ValueError, match="input or target differs"):
        row(a, b, allow_partial=True)


def test_duplicate_observations_are_rejected_even_in_partial_mode():
    a = run([0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        row(replace(a, samples=a.samples * 2), a, allow_partial=True)


@pytest.mark.parametrize("sample_id", [None, "", True, [1]])
def test_missing_and_invalid_ids_are_rejected(sample_id):
    a = run([0])
    with pytest.raises(ValueError, match="invalid sample ID"):
        row(replace(a, samples=[replace(a.samples[0], sample_id=sample_id)]), a)


@pytest.mark.parametrize("epoch", [None, True, 0, -1, 1.5, "1"])
def test_epochs_must_be_positive_integers(epoch):
    a = run([0])
    with pytest.raises(ValueError, match="invalid epoch"):
        row(replace(a, samples=[replace(a.samples[0], epoch=epoch)]), a)


def test_integer_and_string_ids_are_distinct():
    a = run([0, 1])
    b = replace(a, samples=[replace(s, sample_id=str(s.sample_id)) for s in a.samples])
    with pytest.raises(ValueError, match="sets differ"):
        row(a, b)


def test_partial_overlap_is_explicit_and_counted():
    a, b = run([0, 1, 0]), run([1, 1, 1])
    b.samples[0] = replace(b.samples[0], sample_id=3, input="question 3")
    with pytest.raises(ValueError, match="sets differ"):
        row(a, b)
    result = row(a, b, allow_partial=True)
    assert result.samples == 2
    assert (result.baseline_only, result.candidate_only, result.excluded) == (1, 1, 0)


@pytest.mark.parametrize("bad_score", [None, float("nan"), float("inf"), "unknown", [1], {}])
def test_invalid_metrics_are_never_silently_zeroed(bad_score):
    a, b = run([0, 1]), run([bad_score, 1])
    with pytest.raises(ValueError, match="invalid or missing metric"):
        row(a, b)
    result = row(a, b, allow_partial=True)
    assert result.samples == 1
    assert result.excluded == 1


def test_sample_errors_require_explicit_exclusion():
    a, b = run([0, 1]), run([1, 1])
    b.samples[0] = replace(b.samples[0], error=True)
    with pytest.raises(ValueError, match="invalid or missing metric"):
        row(a, b)
    assert row(a, b, allow_partial=True).excluded == 1


def test_incomplete_runs_require_partial_mode():
    a = run([0, 1])
    with pytest.raises(ValueError, match="incomplete runs"):
        row(a, replace(a, status="incomplete"))
    assert row(a, replace(a, status="incomplete"), allow_partial=True).samples == 2


def test_task_mismatches_are_rejected_or_reported():
    a, b = run([0, 1]), run([0, 1])
    a.samples += run([1], task="baseline_extra").samples
    with pytest.raises(ValueError, match="Task sets differ"):
        row(a, b)
    result = compare_runs(a, b, ["accuracy"], n_resamples=99, allow_partial=True)
    assert result.baseline_only_tasks == ["baseline_extra"]


def test_no_common_tasks_or_valid_scores_is_an_error():
    with pytest.raises(ValueError, match="No common tasks"):
        row(run([0], task="a"), run([1], task="b"), allow_partial=True)
    with pytest.raises(ValueError, match="no valid paired"):
        row(run([0]), run([None]), allow_partial=True)


def test_selected_metrics_and_parameters_are_validated():
    a = run([0, 1])
    for metrics in [[], ["accuracy", "accuracy"]]:
        with pytest.raises(ValueError, match="Select at least one metric"):
            compare_runs(a, a, metrics)
    with pytest.raises(ValueError, match="lower_is_better"):
        row(a, a, lower_is_better=["loss"])
    with pytest.raises(ValueError, match="confidence_level"):
        row(a, a, confidence_level=1)


def test_reports_include_changed_samples_and_finite_json():
    result = compare_runs(run([0, 1]), run([1, 1]), ["accuracy"], n_resamples=99)
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert data["comparisons"][0]["changes"][0]["sample_id"] == 0
    markdown = result.to_markdown()
    assert "Adjusted p" in markdown
    assert "paired percentile" in markdown


def test_seed_is_stable_when_another_task_is_added():
    a, b = run([0.2, 0.4, 0.8, 0.2]), run([0.4, 0.5, 0.7, 0.6])
    expected = row(a, b)
    a.samples += run([0, 1], task="extra").samples
    b.samples += run([1, 1], task="extra").samples
    actual = next(r for r in compare_runs(a, b, ["accuracy"], n_resamples=499).comparisons if r.task == "task")
    assert actual.confidence_interval == expected.confidence_interval
    assert actual.p_value == expected.p_value


def test_permutation_p_values_match_independent_exhaustive_sign_flips():
    rng = np.random.default_rng(216)
    for n in range(2, 9):
        for _ in range(5):
            differences = rng.integers(-3, 4, size=n).tolist()
            observed = abs(sum(differences))
            extreme = sum(
                abs(sum(sign * difference for sign, difference in zip(signs, differences))) >= observed
                for signs in product((-1, 1), repeat=n)
            )
            expected = extreme / 2**n
            result = compare_runs(
                run([4] * n), run([4 + d for d in differences]), ["accuracy"], n_resamples=2**n
            ).comparisons[0]
            assert result.test == "paired_permutation"
            assert result.p_value == pytest.approx(expected)


def test_binary_null_calibration():
    rng = np.random.default_rng(819)
    discoveries = 0
    for _ in range(200):
        a = rng.binomial(1, 0.65, size=80).tolist()
        b = rng.binomial(1, 0.65, size=80).tolist()
        result = compare_runs(run(a), run(b), ["accuracy"], n_resamples=19).comparisons[0]
        discoveries += result.adjusted_p_value <= 0.05
    assert discoveries <= 18  # Conservative upper bound for nominal 5% under the null.


def test_bootstrap_coverage_on_paired_gaussian_data():
    rng = np.random.default_rng(821)
    covered = 0
    for _ in range(100):
        a = rng.normal(0, 5, size=80)
        b = a + rng.normal(0.2, 1, size=80)
        result = compare_runs(run(a.tolist()), run(b.tolist()), ["accuracy"], n_resamples=399).comparisons[0]
        covered += result.confidence_interval[0] <= 0.2 <= result.confidence_interval[1]
    assert covered >= 85  # Broad deterministic calibration bound, nominal coverage is 95%.


def write_details(path, scores, **doc_fields):
    logger = DetailsLogger()
    for i, score in enumerate(scores):
        logger.log(
            "task|0",
            Doc(
                **{
                    "query": f"question {i}",
                    "choices": ["gold"],
                    "gold_index": 0,
                    "id": str(i),
                    "task_name": "task",
                    **doc_fields,
                }
            ),
            ModelResponse(text=["answer"]),
            {"accuracy": score},
        )
    pq.write_table(pa.Table.from_pylist([asdict(item) for item in logger.details["task|0"]]), path)


def test_reads_actual_lighteval_details_schema(tmp_path):
    path = tmp_path / "details_task|0_2026-10-16T12-00-00.parquet"
    write_details(path, [0, 1, 0])
    loaded = load_run(path)
    assert loaded.format == "lighteval-parquet"
    assert loaded.samples[0].task == "task|0"
    assert loaded.samples[0].target == ["gold"]
    assert row(loaded, loaded).samples == 3


def test_image_bytes_are_fingerprinted_and_serializable(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    images = [{"bytes": b"saved image", "path": None}]
    write_details(a, [0, 1], images=images)
    write_details(b, [1, 1], images=images)
    report = compare_runs(load_run(a), load_run(b), ["accuracy"], n_resamples=99)
    data = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert len(data["comparisons"][0]["changes"][0]["input"]["images"][0]["bytes"]["sha256"]) == 64


def test_refuses_aggregate_parquet_without_sample_data(tmp_path):
    path = tmp_path / "results.parquet"
    pq.write_table(pa.table({"accuracy": [0.5]}), path)
    with pytest.raises(ValueError, match="not a Lighteval details file"):
        load_run(path)


def test_cli_writes_report_and_reports_pairing_errors(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_details(a, [0] * 10)
    write_details(b, [1] * 10)
    app = typer.Typer()
    app.command()(compare)
    output = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        app, [str(a), str(b), "--metric", "accuracy", "--output", str(output), "--n-resamples", "99"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["comparisons"][0]["verdict"] == "improved"
    result = runner.invoke(app, [str(a), str(b), "--metric", "missing"])
    assert result.exit_code != 0
    assert "invalid or missing metric" in result.output
    result = runner.invoke(app, [str(a), str(b), "--metric", "accuracy", "--output", str(a)])
    assert result.exit_code != 0
    assert "cannot overwrite" in result.output
    result = runner.invoke(
        app, [str(b), str(a), "--metric", "accuracy", "--fail-on-regression", "--n-resamples", "99"]
    )
    assert result.exit_code == 1
    assert "regressed" in result.output


def test_cli_unexpected_failures_exit_with_code_2_not_regression_code(tmp_path, monkeypatch):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_details(a, [0, 1])
    write_details(b, [1, 1])

    def fail(path):
        raise RuntimeError("unreadable log")

    monkeypatch.setattr("lighteval.main_compare.load_run", fail)
    app = typer.Typer()
    app.command()(compare)
    result = CliRunner().invoke(app, [str(a), str(b), "--metric", "accuracy", "--fail-on-regression"])
    assert result.exit_code == 2
    assert "RuntimeError: unreadable log" in result.output


def test_details_with_ids_repeated_across_splits_are_rejected(tmp_path):
    path = tmp_path / "details_task|0_2026-10-16T12-00-00.parquet"
    write_details(path, [0, 1], id="0")
    with pytest.raises(ValueError, match="within each evaluation split"):
        load_run(path)


@pytest.mark.parametrize(
    "field,value",
    [("choices", ["gold", "changed distractor"]), ("images", [{"bytes": b"changed image", "path": None}])],
)
def test_detail_pairing_checks_choices_and_images(tmp_path, field, value):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_details(a, [0, 1])
    write_details(b, [1, 1], **{field: value})
    with pytest.raises(ValueError, match="input or target differs"):
        row(load_run(a), load_run(b))


def test_cli_cannot_overwrite_an_input_through_a_hard_link(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_details(a, [0, 1])
    write_details(b, [1, 1])
    original = a.read_bytes()
    output = tmp_path / "report.json"
    output.hardlink_to(a)
    app = typer.Typer()
    app.command()(compare)
    result = CliRunner().invoke(app, [str(a), str(b), "--metric", "accuracy", "--output", str(output)])
    assert result.exit_code == 2
    assert a.read_bytes() == original


def test_report_write_failure_preserves_existing_file_and_cleans_up(tmp_path, monkeypatch):
    from pathlib import Path

    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_details(a, [0, 1])
    write_details(b, [1, 1])
    output = tmp_path / "report.json"
    output.write_text("existing report")
    original_files = set(tmp_path.iterdir())

    def fail_replace(self, target):
        raise OSError("simulated write failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    app = typer.Typer()
    app.command()(compare)
    result = CliRunner().invoke(app, [str(a), str(b), "--metric", "accuracy", "--output", str(output)])
    assert result.exit_code == 2
    assert "simulated write failure" in result.output
    assert output.read_text() == "existing report"
    assert set(tmp_path.iterdir()) == original_files


def test_report_hard_link_to_directory_input_cannot_modify_source(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    a, b = baseline / "a.parquet", tmp_path / "b.parquet"
    write_details(a, [0, 1])
    write_details(b, [1, 1])
    original = a.read_bytes()
    output = tmp_path / "report.json"
    output.hardlink_to(a)
    app = typer.Typer()
    app.command()(compare)
    result = CliRunner().invoke(app, [str(baseline), str(b), "--metric", "accuracy", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert a.read_bytes() == original
    assert json.loads(output.read_text())["comparisons"][0]["samples"] == 2


def test_reads_real_inspect_logs_with_repeated_epochs(tmp_path):
    image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="

    async def evaluate():
        dataset = [
            Sample(
                input=[ChatMessageUser(content=[ContentText(text=f"question {i}"), ContentImage(image=image)])],
                target="yes",
                id=str(i),
            )
            for i in range(3)
        ]
        model = get_model(
            "mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "yes")] * 6, memoize=False
        )
        logs = await eval_async(
            Task(name="inspect_task", dataset=dataset, solver=generate(cache=False), scorer=exact()),
            model=model,
            epochs=2,
            max_samples=1,
            log_dir=str(tmp_path),
        )
        return logs[0].location

    log_path = asyncio.run(evaluate())
    loaded = load_run(log_path)
    assert loaded.samples[0].input[0]["content"][1]["image"] == image
    result = compare_runs(loaded, loaded, ["exact"], n_resamples=99).comparisons[0]
    assert result.samples == 3
    assert result.observations == 6
    assert result.baseline_mean == 1
    assert result.confidence_interval == (0, 0)
    assert len(loaded.metadata["files"][0]["sha256"]) == 64
    assert loaded.metadata["files"][0]["task"] == "inspect_task"
    json_path = tmp_path / "native.json"
    write_eval_log(read_eval_log(log_path), json_path)
    json_run = load_run(json_path)
    assert compare_runs(loaded, json_run, ["exact"], n_resamples=99).comparisons[0].delta == 0
    dictionary_log = read_eval_log(log_path)
    for sample in dictionary_log.samples:
        sample.scores["exact"].value = {"accuracy": sample.scores["exact"].value}
    dictionary_path = tmp_path / "dictionary.json"
    write_eval_log(dictionary_log, dictionary_path)
    dictionary_run = load_run(dictionary_path)
    assert (
        compare_runs(dictionary_run, dictionary_run, ["exact.accuracy"], n_resamples=99).comparisons[0].baseline_mean
        == 1
    )
    collision_log = dictionary_log.model_copy(deep=True)
    for sample in collision_log.samples:
        sample.scores["exact.accuracy"] = sample.scores["exact"].model_copy(update={"value": 0})
    collision_path = tmp_path / "collision.json"
    write_eval_log(collision_log, collision_path)
    with pytest.raises(ValueError, match="Ambiguous metric"):
        load_run(collision_path)
    collision_path.unlink()
    dictionary_log.eval.model = "mockllm/other-model"
    write_eval_log(dictionary_log, tmp_path / "other-model.json")
    with pytest.raises(ValueError, match="different models"):
        load_run(tmp_path)


def test_serializes_changed_conversational_inputs():
    a = run([0, 0])
    a.samples = [replace(s, input=[ChatMessageUser(content=s.input)]) for s in a.samples]
    b = replace(a, samples=[replace(s, scores={"accuracy": 1}) for s in a.samples])
    report = compare_runs(a, b, ["accuracy"], n_resamples=99)
    data = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert data["comparisons"][0]["changes"][0]["input"][0]["role"] == "user"


def test_independent_message_ids_do_not_prevent_pairing():
    a, b = run([0, 1]), run([1, 1])
    a.samples = [replace(s, input=[ChatMessageUser(content=s.input)]) for s in a.samples]
    b.samples = [replace(s, input=[ChatMessageUser(content=s.input)]) for s in b.samples]
    assert a.samples[0].input[0].id != b.samples[0].input[0].id
    assert row(a, b).samples == 2
    b.samples[0] = replace(b.samples[0], input=[ChatMessageUser(content="different question")])
    with pytest.raises(ValueError, match="input or target differs"):
        row(a, b)

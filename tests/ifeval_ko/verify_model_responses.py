"""Compare selected or full-split saved responses with original Ko and the registered task.

The run, reference evidence and original sources are read-only. No model weights
are loaded, no responses are regenerated and no scoring rules are implemented.
"""

import argparse
import ast
import contextlib
import copy
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

from verify_registered_task import (
    AUDIT_REVISION,
    KO_REVISION,
    REPO,
    capture,
    format_doc,
    git_head,
    load_reference,
    load_task,
)
from verify_scoring import KINDS, compare_flags, counts, read, require, sha


SCOPES = {"selected": (23, 40), "full": (342, 481)}
ROW_FIELDS = ("key", "prompt", "instruction_id_list", "kwargs")
ENGINES = ("original", "direct_ported", "registered")
METRICS = tuple(f"{level}_level_{mode}_acc" for mode in KINDS for level in ("prompt", "inst"))
RUN_FILES = {
    "generate_responses.py",
    "manifest.json",
    "requirements.lock.txt",
    "responses.jsonl",
    "run.log",
    "selected-samples.json",
    "selection.json",
}


def object_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def agreed_keys():
    """Read the merged generator's selection contract without executing it."""
    source = REPO / "scripts/ifeval_ko/generate_responses.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KEYS" for t in node.targets):
            keys = ast.literal_eval(node.value)
            require(len(keys) == len(set(keys)) == SCOPES["selected"][0], "Invalid generator selection contract")
            return keys
    raise ValueError("Generator selection contract not found")


def check_run_hashes(directory):
    checksums = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        require(name in RUN_FILES and name not in checksums, f"Unexpected or duplicate checksum entry: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"Invalid checksum: {name}")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"Missing or linked run file: {name}")
        require(sha(path) == digest, f"Run checksum mismatch: {name}")
        checksums[name] = digest
    require(set(checksums) == RUN_FILES, "Run checksums must cover all seven payload files")
    return checksums


def check_archive(directory, archive, expected_sha256):
    if archive is None:
        return {"status": "NOT_PROVIDED"}
    require(sha(archive) == expected_sha256, "Archive SHA-256 mismatch")
    wanted, found = RUN_FILES | {"SHA256SUMS"}, set()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = Path(entry.filename)
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe archive member path")
            if entry.is_dir() or path.name not in wanted:
                continue
            require(path.name not in found, f"Duplicate archive payload: {path.name}")
            require(bundle.read(entry) == (directory / path.name).read_bytes(), f"Archive/run mismatch: {path.name}")
            found.add(path.name)
    require(found == wanted, "Archive payload files are missing")
    return {"status": "PASS", "sha256": expected_sha256, "payload_files": len(found)}


def raw_input(record):
    return {field: copy.deepcopy(record[field]) for field in ROW_FIELDS}


def check_record(record):
    key = record["key"]
    require(type(key) is int, "Source key must be an integer")
    require(record["status"] == "generated", f"Response is not successfully generated: {key}")
    response = record["response"]
    require(type(response) is str and record["scoring_response"] == response, f"Response contract differs: {key}")
    require(hashlib.sha256(response.encode("utf-8")).hexdigest() == record["response_sha256"], f"Response hash: {key}")
    ids, kwargs = record["instruction_id_list"], record["kwargs"]
    require(isinstance(ids, list) and ids and all(type(iid) is str for iid in ids), f"Instruction IDs: {key}")
    require(isinstance(kwargs, list) and len(ids) == len(kwargs), f"Instruction/kwargs alignment: {key}")
    require(all(type(value) is dict for value in kwargs), f"Invalid kwargs: {key}")
    tokens = record["output_token_ids"]
    require(
        isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens), f"Token IDs: {key}"
    )
    require(
        type(record["generated_token_count"]) is int and record["generated_token_count"] == len(tokens),
        f"Token count: {key}",
    )
    require(record["finish_reason"] in {"eos", "stop_string", "length", "unknown"}, f"Finish reason: {key}")


def check_manifest(manifest, records, scope="selected"):
    expected_prompts, expected_instructions = SCOPES[scope]
    for field, expected in {
        "expected_count": expected_prompts,
        "generated_count": expected_prompts,
        "generation_error_count": 0,
        "unattempted_count": 0,
        "instruction_occurrences": expected_instructions,
    }.items():
        require(type(manifest[field]) is int and manifest[field] == expected, f"Manifest count differs: {field}")
    logging_only = manifest.get("error_stage") == "logging" and bool(manifest.get("logging_errors"))
    require(
        manifest["status"] == "responses_generated" or (manifest["status"] == "error" and logging_only),
        "Run has an incomplete or unexplained failure",
    )
    require(
        manifest["dataset_id"] == "allganize/IFEval-Ko" and manifest["dataset_revision"] == KO_REVISION,
        "Dataset revision differs",
    )
    require(
        manifest["dataset_subset"] == "default" and manifest["dataset_split"] == "train",
        "Dataset subset/split differs",
    )
    require(manifest["num_fewshots"] == 0, "Expected zero-shot generation")
    require(manifest.get("evaluation_scope", "selected") == scope, "Manifest evaluation scope differs")
    require(manifest["selected_keys"] == [row["key"] for row in records], "Manifest selection differs")
    require(
        manifest["empty_response_count"] == sum(not row["response"].strip() for row in records),
        "Empty-response count differs",
    )
    require(
        manifest["length_limited_count"] == sum(row["finish_reason"] == "length" for row in records),
        "Length-limit count differs",
    )


def load_run(directory, archive=None, archive_sha256=None, scope="selected"):
    expected_prompts, expected_instructions = SCOPES[scope]
    checksums = check_run_hashes(directory)
    archive_info = check_archive(directory, archive, archive_sha256)
    manifest, selection = read(directory / "manifest.json"), read(directory / "selection.json")
    recorded = manifest["artifact_sha256"]
    require(
        recorded == {name: digest for name, digest in checksums.items() if name != "manifest.json"},
        "Manifest artifact hashes differ",
    )
    selected = read(directory / "selected-samples.json")
    with (directory / "responses.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    keys = [row["key"] for row in records]
    require(len(keys) == len(set(keys)) == expected_prompts, "Missing or extra prompt records or duplicate keys")
    require(selection.get("scope", "selected") == scope, "Selection evaluation scope differs")
    if scope == "selected":
        require(selection["keys"] == agreed_keys(), "Selection differs from the agreed 23 keys/order")
    require(len(records) == len(selected) == expected_prompts, "Missing or extra prompt records")
    require(
        keys == [row["key"] for row in selected] == selection["keys"],
        "Duplicate, missing or reordered keys",
    )
    for record, source in zip(records, selected):
        check_record(record)
        require(
            object_hash(raw_input(record)) == object_hash(raw_input(source)), f"Saved input differs: {record['key']}"
        )
    require(
        sum(len(row["instruction_id_list"]) for row in records) == expected_instructions,
        f"Expected {expected_instructions} instructions",
    )
    require(len({iid for row in records for iid in row["instruction_id_list"]}) == 21, "Expected 21 instruction types")
    check_manifest(manifest, records, scope)
    return {
        "records": records,
        "manifest": manifest,
        "integrity": {"status": "PASS", "run_sha256": checksums, "archive": archive_info},
    }


def check_source_rows(records, source_rows, scope="selected"):
    require(
        len(source_rows) == 342 and sum(len(row["instruction_id_list"]) for row in source_rows) == 481,
        "Unexpected pinned dataset size",
    )
    indexed = {row["key"]: row for row in source_rows}
    require(len(indexed) == 342, "Duplicate pinned dataset keys")
    expected_keys = [row["key"] for row in source_rows] if scope == "full" else agreed_keys()
    require([row["key"] for row in records] == expected_keys, "Pinned source key order or completeness differs")
    for record in records:
        require(
            object_hash(raw_input(record)) == object_hash(raw_input(indexed[record["key"]])),
            f"Pinned source input differs: {record['key']}",
        )


def check_decision(value, instruction_count):
    require(type(value) is dict and set(value) == set(METRICS), "Expected the four metric fields")
    for mode in KINDS:
        flags, prompt = value[f"inst_level_{mode}_acc"], value[f"prompt_level_{mode}_acc"]
        require(type(flags) is list and len(flags) == instruction_count, f"Wrong {mode} instruction denominator")
        require(all(type(flag) is bool for flag in flags) and type(prompt) is bool, f"Non-Boolean {mode} judgment")
        require(prompt == all(flags), f"Inconsistent {mode} prompt judgment")


def score_registered(task, response_class, raw, response):
    doc = format_doc(task, raw)
    supplied = {
        "key": doc.specific["ifeval_ko"]["key"],
        "prompt": doc.query,
        "instruction_id_list": doc.specific["instructions_id_list"],
        "kwargs": doc.specific["kwargs"],
    }
    require(object_hash(supplied) == object_hash(raw), "Registered Doc changed the source fields")
    fields = ("query", "choices", "gold_index", "instruction", "specific")
    before = object_hash({field: getattr(doc, field) for field in fields})
    model_response = response_class(text=[response])
    require(model_response.final_text == [response], "ModelResponse changed the saved response")
    value = task.metrics[0].compute_sample(doc=doc, model_response=model_response)
    require(
        object_hash({field: getattr(doc, field) for field in fields}) == before, "Scoring mutated the registered Doc"
    )
    return value


def reset_seeds():
    from langdetect import DetectorFactory

    random.seed(0)
    DetectorFactory.seed = 0
    return {"python": 0, "langdetect": DetectorFactory.seed, "python_state_sha256": object_hash(random.getstate())}


def checked_score(fn, raw, response, seeds):
    supplied = copy.deepcopy(raw)
    before = object_hash(supplied)
    execution = {
        "input_sha256": before,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "seeds": seeds,
    }
    result = capture(fn, supplied, response)
    result["execution"] = execution
    if "error" in result:
        return result
    try:
        require(object_hash(supplied) == before, "Scoring mutated the source input")
        check_decision(result["value"], len(raw["instruction_id_list"]))
    except Exception as error:
        return {
            "error": {"type": type(error).__name__, "message": str(error)},
            "returned_value": repr(result["value"]),
            "execution": execution,
        }
    return result


def compare_model_metrics(rows, task, mismatches):
    totals = {engine: counts([row[engine]["value"] for row in rows]) for engine in ENGINES}
    aggregation, metrics = capture(task.aggregation), {}
    for name in METRICS:
        detail = {engine: totals[engine][name] for engine in ENGINES}
        if "error" in aggregation:
            detail.update(passed=False, error=aggregation["error"])
        else:
            try:
                native = float(
                    aggregation["value"][name](copy.deepcopy([row["registered"]["value"][name] for row in rows]))
                )
                require(math.isfinite(native), "Non-finite native aggregation")
                error = abs(native - detail["original"]["value"])
                passed = detail["original"] == detail["direct_ported"] == detail["registered"] and error <= 1e-12
                detail.update(native_value=native, absolute_error=error, passed=passed)
            except Exception as error:
                detail.update(passed=False, error={"type": type(error).__name__, "message": str(error)})
        metrics[name] = detail
        if not detail["passed"]:
            mismatches.append({"source": "corpus aggregation", "metric": name, **detail})
    return metrics


def matched_decisions(rows):
    result = {mode: {"prompts": 0, "instructions": 0} for mode in KINDS}
    for row in rows:
        if any("error" in row[engine] for engine in ENGINES):
            continue
        values = [row[engine]["value"] for engine in ENGINES]
        for mode in KINDS:
            prompts = [value[f"prompt_level_{mode}_acc"] for value in values]
            result[mode]["prompts"] += int(prompts[0] == prompts[1] == prompts[2])
            result[mode]["instructions"] += sum(entry[mode]["matched"] for entry in row["instruction_comparisons"])
    return result


def instruction_comparisons(row):
    comparisons = []
    for index, instruction_id in enumerate(row["instruction_ids"]):
        entry = {"key": row["key"], "instruction_index": index, "instruction_id": instruction_id}
        for mode in KINDS:
            values = {}
            for engine in ENGINES:
                flags = row[engine].get("value", {}).get(f"inst_level_{mode}_acc", [])
                values[engine] = flags[index] if index < len(flags) else None
            votes = [values[engine] for engine in ENGINES]
            values["matched"] = all(type(flag) is bool for flag in votes) and votes[0] == votes[1] == votes[2]
            entry[mode] = values
        comparisons.append(entry)
    return comparisons


def score_record(record, engines, mismatches):
    raw, response = raw_input(record), record["response"]
    row = {
        "key": raw["key"],
        "instruction_ids": raw["instruction_id_list"],
        "input_sha256": object_hash(raw),
        "response_sha256": record["response_sha256"],
        "finish_reason": record["finish_reason"],
    }
    start = len(mismatches)
    for engine, fn in engines.items():
        row[engine] = checked_score(fn, raw, response, reset_seeds())
        if "error" in row[engine]:
            mismatches.append({"key": raw["key"], "source": engine, "error": row[engine]["error"]})
    if not any("error" in row[engine] for engine in ENGINES):
        case = {
            "case_id": f"model:{raw['key']}",
            "key": raw["key"],
            "ko_instruction_id_list": raw["instruction_id_list"],
        }
        for engine in ("direct_ported", "registered"):
            compare_flags(case, row[engine]["value"], row["original"]["value"], mismatches, engine)
    row["instruction_comparisons"] = instruction_comparisons(row)
    row["passed"] = len(mismatches) == start
    return row


def score_records(records, reference, task, scoring, response_class, progress=None, scope="selected"):
    expected_prompts, expected_instructions = SCOPES[scope]
    keys = [row["key"] for row in records]
    require(
        len(records) == len(set(keys)) == expected_prompts and (scope == "full" or keys == agreed_keys()),
        f"Scoring requires all {expected_prompts} unique prompts in order",
    )
    require(
        sum(len(row["instruction_id_list"]) for row in records) == expected_instructions,
        f"Scoring requires all {expected_instructions} instructions",
    )
    engines = {
        "original": lambda raw, text: reference.process_results(raw, [text]),
        "direct_ported": lambda raw, text: scoring.score_response(
            raw["prompt"], raw["instruction_id_list"], raw["kwargs"], text
        ),
        "registered": lambda raw, text: score_registered(task, response_class, raw, text),
    }
    rows, mismatches = [], []
    if progress is not None:
        progress.update(results=rows, mismatches=mismatches)
    for record in records:
        if progress is not None:
            progress["active_input"] = {**raw_input(record), "response": record["response"]}
        rows.append(score_record(record, engines, mismatches))
        if progress is not None and mismatches and progress.get("first_mismatch") is None:
            progress["first_mismatch"] = {**copy.deepcopy(mismatches[0]), "input": progress["active_input"]}
    if progress is not None:
        progress.pop("active_input", None)
    errors = sum("error" in row[engine] for row in rows for engine in ENGINES)
    metrics = compare_model_metrics(rows, task, mismatches) if not errors else {}
    aggregation_errors = sum("error" in metric for metric in metrics.values())
    first = copy.deepcopy(mismatches[0]) if mismatches else None
    if first is not None and "key" in first:
        record = next(record for record in records if record["key"] == first["key"])
        first["input"] = {**raw_input(record), "response": record["response"]}
    elif first is not None:
        first["input"] = {
            "scope": "corpus",
            "records": [{**raw_input(record), "response": record["response"]} for record in records],
        }
    return {
        "results": rows,
        "matched_cases": sum(row["passed"] for row in rows),
        "matched_decisions": matched_decisions(rows),
        "scoring_errors": errors,
        "aggregation_errors": aggregation_errors,
        "metrics_status": "COMPLETE" if not errors and not aggregation_errors else "INCOMPLETE",
        "metrics": metrics,
        "mismatches": mismatches,
        "first_mismatch": first,
        "passed": not mismatches,
    }


def environment_info():
    packages = {}
    for name in (
        "lighteval",
        "torch",
        "transformers",
        "numpy",
        "datasets",
        "nltk",
        "langdetect",
        "immutabledict",
        "packaging",
        "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "target_sha": git_head(),
        "worktree_status": subprocess.check_output(["git", "-C", str(REPO), "status", "--short"], text=True).strip(),
        "verifier_sha256": sha(Path(__file__)),
        "helper_sha256": {
            name: sha(Path(__file__).with_name(name)) for name in ("verify_registered_task.py", "verify_scoring.py")
        },
        "package_versions": packages,
        "audit_revision": AUDIT_REVISION,
        "ko_revision": KO_REVISION,
        "random_seed": 0,
        "langdetect_seed": 0,
        "workers": 1,
        "reference_import_mode": "isolated-source",
    }


def run(args, report):
    report["stage"] = "input"
    print("Validating the saved run, archive checksums, response hashes and selected keys.")
    saved = load_run(args.run_dir, args.archive, args.archive_sha256, args.scope)
    records = saved["records"]
    report.update(integrity=saved["integrity"], generation_manifest=saved["manifest"])
    report["stage"] = "loading evaluators"
    print("Loading hash-pinned original Ko and validating the source dataset.")
    reference, _, evidence_hashes, sources = load_reference(args.audit_dir, args.source_root)
    import pyarrow.parquet as pq

    source = args.source_root / "sources/ko/data/train-00000-of-00001.parquet"
    check_source_rows(records, pq.read_table(source).to_pylist(), args.scope)
    report["integrity"].update(pinned_source_inputs="PASS", dataset_sha256=sha(source))
    task, scoring, response_class, code_hashes = load_task()
    report["environment"].update(
        task_source_sha256=code_hashes, evidence_sha256=evidence_hashes, verified_sources=sources
    )
    if getattr(reference, "__file__", None):
        report["environment"]["reference_runtime"] = {
            "module": reference.__name__,
            "path": str(Path(reference.__file__).resolve()),
            "sha256": sha(Path(reference.__file__)),
            "callable": reference.process_results.__qualname__,
        }
    report["environment"]["registered_task"] = task.name
    report["stage"] = "scoring"
    print(
        f"Replaying {len(records)} identical saved responses through original Ko, direct ported scorer and registered task."
    )
    report.update(score_records(records, reference, task, scoring, response_class, progress=report, scope=args.scope))
    report["length_limited_keys"] = [record["key"] for record in records if record["finish_reason"] == "length"]
    report["status"] = "PASS" if report["passed"] else "FAIL"
    report["stage"] = "complete"
    print(
        f"{report['status']}: {report['matched_cases']}/{report['expected_prompts']} cases agree; scoring errors={report['scoring_errors']}, aggregation errors={report['aggregation_errors']}, mismatches={len(report['mismatches'])}."
    )


def render_markdown(report):
    prompts, instructions = report["expected_prompts"], report["expected_instructions"]
    lines = [
        "# IFEval-Ko saved-model-response comparison",
        "",
        f"Result: **{report['status']}**",
        "",
        f"Scope: {report['evaluation_scope']} {prompts} prompts / {instructions} instructions; saved responses.",
        "Native Lighteval model inference: **not run**.",
        "",
        "Agreement measures evaluator compatibility; it is not a claim of 100% model accuracy.",
        "",
    ]
    for mode, values in report.get("matched_decisions", {}).items():
        lines.append(
            f"- {mode}: {values['prompts']}/{prompts} prompt and {values['instructions']}/{instructions} instruction judgments agree across all three paths."
        )
    lines += [
        "",
        f"Scoring errors: {report.get('scoring_errors', 'not measured')}",
        f"Aggregation errors: {report.get('aggregation_errors', 'not measured')}",
        f"Metric comparison: {report.get('metrics_status', 'not run')}",
        f"Retained token-limit keys: {report.get('length_limited_keys', [])}",
        "",
        "| Metric | Original Ko | Ported scorer | Registered task | Native absolute error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, detail in report.get("metrics", {}).items():
        values = [
            f"{detail[engine]['numerator']}/{detail[engine]['denominator']} ({detail[engine]['value']:.8f})"
            for engine in ENGINES
        ]
        lines.append(f"| `{name}` | {' | '.join(values)} | {detail.get('absolute_error', 'error')} |")
    if report.get("metrics_status") != "COMPLETE":
        lines += [
            "",
            f"Expected denominators remain {prompts} prompts and {instructions} instructions. Failed or missing scoring produces no aggregate scores; native aggregation errors are recorded per metric.",
        ]
    lines += [
        "",
        "## Provenance",
        "",
        f"Evaluation code HEAD: `{report['environment']['target_sha']}`",
        f"Evaluation verifier SHA-256: `{report['environment']['verifier_sha256']}`",
    ]
    saved = report.get("generation_manifest", {})
    hashes = report.get("integrity", {}).get("run_sha256", {})
    lines += [
        f"Generation code revision: `{saved.get('code_revision', 'unavailable')}`",
        f"Saved generator snapshot SHA-256: `{hashes.get('generate_responses.py', 'unavailable')}`",
        f"Archive SHA-256: `{report.get('integrity', {}).get('archive', {}).get('sha256', 'not provided')}`",
        "",
        "Generation and evaluation revisions are recorded separately. Original inputs and responses are unchanged.",
        "",
    ]
    diagnostic = report.get("error") or report.get("first_mismatch")
    if diagnostic:
        lines += ["## First failure", "", "```json", json.dumps(diagnostic, ensure_ascii=False, indent=2), "```", ""]
    lines += [
        "Complete decisions, all mismatches, source hashes, versions and the command are in `model-response-parity.json`.",
        "",
    ]
    return "\n".join(lines)


def write_reports(directory, report):
    (directory / "model-response-parity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "model-response-parity.md").write_text(render_markdown(report), encoding="utf-8")
    names = ("model-response-parity.json", "model-response-parity.md", "run.log")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha(directory / name)}  {name}\n" for name in names), encoding="utf-8"
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=SCOPES, default="selected", help="Selected 23 prompts or the full 342-prompt split"
    )
    for name in ("run-dir", "audit-dir", "source-root", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--archive", type=Path, help="Optional original ZIP; compared byte-for-byte with the run payload"
    )
    parser.add_argument("--archive-sha256", help="Trusted published SHA-256 of --archive")
    args = parser.parse_args(argv)
    if bool(args.archive) != bool(args.archive_sha256):
        parser.error("--archive and --archive-sha256 must be supplied together")
    if args.archive_sha256 and re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256) is None:
        parser.error("--archive-sha256 must be a lowercase SHA-256 digest")
    for name in ("run_dir", "audit_dir", "source_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        parser.error("--output-dir already exists; use a new directory to preserve previous results")
    if any(args.output_dir.is_relative_to(root) for root in (args.run_dir, args.audit_dir, args.source_root)):
        parser.error("--output-dir must be outside all read-only input directories")
    return args


def main(argv=None):
    args = parse_args(argv)
    expected_prompts, expected_instructions = SCOPES[args.scope]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    os.environ.setdefault("HF_HOME", str(args.output_dir / ".cache/huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / ".cache"))
    report = {
        "status": "RUNNING",
        "passed": False,
        "stage": "starting",
        "command": [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])],
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": f"{expected_prompts} saved responses; original Ko vs ported scorer vs registered task; no model inference",
        "evaluation_scope": args.scope,
        "expected_prompts": expected_prompts,
        "expected_instructions": expected_instructions,
        "metrics_status": "NOT_RUN",
        "metrics": {},
        "mismatches": [],
        "first_mismatch": None,
        "environment": environment_info(),
    }
    with (args.output_dir / "run.log").open("x", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(logfile), contextlib.redirect_stderr(logfile):
            try:
                run(args, report)
            except (Exception, KeyboardInterrupt) as error:
                report.update(
                    status="ERROR", passed=False, error={"type": type(error).__name__, "message": str(error)}
                )
                if report["stage"] == "scoring":
                    report["metrics_status"] = "INCOMPLETE"
                active = report.pop("active_input", None)
                if active is not None:
                    report["execution_error_input"] = active
                    if report["first_mismatch"] is None:
                        first = (
                            copy.deepcopy(report["mismatches"][0])
                            if report["mismatches"]
                            else {"key": active["key"], "source": "execution", "error": report["error"]}
                        )
                        report["first_mismatch"] = {**first, "input": active}
                traceback.print_exc()
    report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_reports(args.output_dir, report)
    with contextlib.suppress(OSError):
        print(f"{report['status']}: {args.output_dir / 'model-response-parity.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

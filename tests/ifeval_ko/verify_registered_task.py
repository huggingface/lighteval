"""Replay the pinned IFEval-Ko audit through the registered task without a model.

Supply read-only audit/source directories and a new output path. Optional saved
scorer or registered-task results are compared by case_id and instruction index.
"""

import argparse
import copy
import datetime
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import traceback
from pathlib import Path

from verify_scoring import (
    KINDS,
    compare_flags,
    counts,
    from_reference,
    index_records,
    load_cases,
    oracle,
    read,
    require,
    sha,
)


REPO = Path(__file__).resolve().parents[2]
KO_REVISION = "54199e3801116897697babf341865741dcd06fc8"
AUDIT_REVISION = "da7053db1d4858239856695a2cfed702173234b6"
CODE_NAMES = ("main", "scoring", "instructions", "instructions_registry", "instructions_util")


def capture(fn, *args, **kwargs):
    try:
        return {"value": fn(*args, **kwargs)}
    except Exception as error:
        return {"error": {"type": type(error).__name__, "message": str(error)}}


def load_reference(audit, source_root):
    """Verify evidence bytes before importing its unchanged reference loader."""
    evidence_hashes = {}
    for line in (audit / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        require(sha(audit / filename) == digest, f"Evidence hash mismatch: {filename}")
        evidence_hashes[filename] = digest
    os.environ["NLTK_DATA"] = str(source_root / ".cache/nltk_data")
    os.environ["IFEVAL_AUDIT_SOURCE_ROOT"] = str(source_root)
    sys.path.insert(0, str(audit))
    import audit_runtime
    import case_utils
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    audit_runtime.SOURCE_ROOT = source_root
    provenance, verified = case_utils.verify_sources(source_root)
    require(provenance["ko"]["revision"] == KO_REVISION, "Wrong Ko reference revision")
    resources = read(audit / "nltk_resources.json")["sha256"]
    for filename, digest in resources.items():
        require(sha(source_root / ".cache/nltk_data" / filename) == digest, f"NLTK resource mismatch: {filename}")
    modules = audit_runtime.load_modules("ko", "isolated-source")
    for name, module in modules.items():
        expected = provenance["ko"]["files"][f"ifeval_ko/{name}.py"]["sha256"]
        require(sha(Path(module.__file__)) == expected, f"Reference module mismatch: {name}")
    return modules["utils"], case_utils, evidence_hashes, verified


def load_task():
    sys.path.insert(0, str(REPO / "src"))
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.tasks.ifeval_ko import main, scoring

    code_hashes = {}
    for name in CODE_NAMES:
        module = importlib.import_module("lighteval.tasks.tasks.ifeval_ko." + name)
        source = REPO / "src/lighteval/tasks/tasks/ifeval_ko" / (name + ".py")
        require(sha(Path(module.__file__)) == sha(source), f"Loaded code differs from this checkout: {name}")
        code_hashes[name + ".py"] = sha(source)
    task = Registry(tasks="ifeval_ko|0").load_tasks()["ifeval_ko|0"]
    require(task.formatter is main.ifeval_ko_prompt, "Unexpected registered formatter")
    require(task.dataset_path == "allganize/IFEval-Ko" and task.dataset_revision == KO_REVISION, "Wrong dataset")
    require(task.dataset_config_name == "default" and task.evaluation_split == ["train"], "Wrong subset/split")
    require(task.config.num_fewshots == 0, "Expected zero-shot evaluation")
    require(len(task.metrics) == 1 and set(task.metrics[0].metric_name) == set(main.METRIC_NAMES), "Wrong metrics")
    return task, scoring, ModelResponse, code_hashes


def format_doc(task, raw):
    doc = task.formatter(copy.deepcopy(raw), task.name)
    require(doc.query == raw["prompt"], "Doc.query changed")
    require(doc.specific["instructions_id_list"] == raw["instruction_id_list"], "Doc instruction order changed")
    require(doc.specific["kwargs"] == raw["kwargs"], "Doc kwargs changed")
    require(doc.specific["ifeval_ko"]["key"] == raw["key"], "Doc key changed")
    require(doc.choices == [""] and doc.gold_index == 0 and doc.instruction == "", "Doc formatting changed")
    return doc


def compute_registered(task, raw, response_cls, text):
    doc = format_doc(task, raw)
    result = task.metrics[0].compute_sample(doc=doc, model_response=response_cls(text=[text]))
    require(doc.query == raw["prompt"] and doc.specific["kwargs"] == raw["kwargs"], "Scoring mutated Doc input")
    return result


def compare_result(case, actual, expected, label, mismatches):
    try:
        compare_flags(case, actual, expected, mismatches, label)
    except (AttributeError, KeyError, TypeError) as error:
        mismatches.append(
            {
                "case_id": case["case_id"],
                "key": case["key"],
                "source": label,
                "reference": expected,
                "new": actual,
                "error": str(error),
            }
        )


def load_external(path, label, cases, support):
    if path is None:
        return None
    document = read(path)
    require(document.get("passed") is True, f"Saved {label} run did not pass")
    if label == "direct scorer":
        require(
            document.get("environment", {}).get("checker_mode") == "ported",
            "Saved direct run must use --checker-mode ported",
        )
    normal = index_records(document["results"], 156)
    auxiliary = index_records(document["support_results"], 18)
    require(set(normal) == {case["case_id"] for case in cases}, f"{label} case IDs differ")
    require(set(auxiliary) == {case["case_id"] for case in support}, f"{label} support case IDs differ")
    for case in cases:
        row = normal[case["case_id"]]
        expected = {
            "key": case["key"],
            "instruction_ids": case["ko_instruction_id_list"],
            "input_sha256": case["hashes"]["input_sha256"],
            "response_sha256": case["hashes"]["response_sha256"],
        }
        require(
            all(row.get(key) == value for key, value in expected.items()), f"{label} input mismatch: {case['case_id']}"
        )
    return {"document": document, "normal": normal, "support": auxiliary, "sha256": sha(path)}


def validate_normal(cases, stored, task, scoring, response_cls, reference, case_utils, external, mismatches):
    rows = []
    for case in cases:
        raw = copy.deepcopy(case_utils.ko_input(case))
        before = copy.deepcopy(raw)
        response = case["test_response_ko"]
        engines = {
            "original": lambda: reference.process_results(raw, [response]),
            "direct_ported": lambda: scoring.score_response(
                raw["prompt"], raw["instruction_id_list"], raw["kwargs"], response
            ),
            "registered": lambda: compute_registered(task, raw, response_cls, response),
        }
        row = {
            "case_id": case["case_id"],
            "key": case["key"],
            "instruction_ids": raw["instruction_id_list"],
            "input_sha256": case["hashes"]["input_sha256"],
            "response_sha256": case["hashes"]["response_sha256"],
        }
        start = len(mismatches)
        for label, fn in engines.items():
            random.seed(0)
            result = capture(fn)
            row[label] = result
            if "error" in result:
                mismatches.append(
                    {"case_id": case["case_id"], "key": case["key"], "source": label, "error": result["error"]}
                )
            else:
                compare_result(case, result["value"], oracle(case), label + " vs oracle", mismatches)
                compare_result(
                    case, result["value"], from_reference(stored[case["case_id"]]), label + " vs stored Ko", mismatches
                )
                for name, saved in external.items():
                    if saved is not None:
                        compare_result(
                            case,
                            result["value"],
                            saved["normal"][case["case_id"]]["value"],
                            label + " vs " + name,
                            mismatches,
                        )
            require(raw == before, f"{label} changed fixture input: {case['case_id']}")
        row["passed"] = len(mismatches) == start
        row.update(row["registered"])
        rows.append(row)
    return rows


def compare_support(case, result, mode, label, mismatches):
    expected = case["expected_ko"]
    wanted_error = expected.get("error_type")
    actual_error = result.get("error", {}).get("type")
    if actual_error != wanted_error:
        mismatches.append(
            {
                "case_id": case["case_id"],
                "key": case["key"],
                "instruction_index": case["instruction_index"],
                "instruction_id": case["instruction_id"],
                "mode": mode,
                "source": label,
                "reference_error": wanted_error,
                "new": result,
            }
        )
    elif wanted_error is None:
        require(
            type(result["prompt_result"]) is bool
            and all(type(flag) is bool for flag in result["instruction_results"]),
            "Non-Boolean support result",
        )
        if (
            result["instruction_results"] != expected["full"][mode]
            or result["prompt_result"] != expected["prompt"][mode]
        ):
            mismatches.append(
                {
                    "case_id": case["case_id"],
                    "key": case["key"],
                    "instruction_index": case["instruction_index"],
                    "instruction_id": case["instruction_id"],
                    "mode": mode,
                    "source": label,
                    "reference": expected,
                    "new": result,
                }
            )


def validate_support(cases, task, scoring, response_cls, reference, case_utils, external, mismatches):
    rows = []
    for case in cases:
        raw = copy.deepcopy(case_utils.ko_input(case))
        text = case["test_response_ko"]
        inp = reference.InputExample(
            key=raw["key"], prompt=raw["prompt"], instruction_id_list=raw["instruction_id_list"], kwargs=raw["kwargs"]
        )
        phases = {}
        start = len(mismatches)
        for engine in ("original", "ported"):
            random.seed(0)
            phases[engine] = {}
            for mode in KINDS:
                if engine == "original":
                    result = capture(getattr(reference, "test_instruction_following_" + mode), inp, text)
                else:
                    result = capture(
                        getattr(scoring, "score_" + mode),
                        raw["prompt"],
                        raw["instruction_id_list"],
                        raw["kwargs"],
                        text,
                    )
                if "value" in result:
                    value = result.pop("value")
                    result.update(
                        instruction_results=value.follow_instruction_list, prompt_result=value.follow_all_instructions
                    )
                phases[engine][mode] = result
                compare_support(case, result, mode, engine, mismatches)
        random.seed(0)
        combined = capture(compute_registered, task, raw, response_cls, text)
        for mode in KINDS:
            if "error" in combined:
                result = combined
            else:
                result = {
                    "instruction_results": combined["value"][f"inst_level_{mode}_acc"],
                    "prompt_result": combined["value"][f"prompt_level_{mode}_acc"],
                }
            compare_support(case, result, mode, "registered combined", mismatches)
        for label, saved in external.items():
            if saved is None:
                continue
            old = saved["support"][case["case_id"]]["combined"]
            require(
                combined.get("error", {}).get("type") == old.get("error", {}).get("type"),
                f"{label} support exception differs",
            )
            require(combined.get("value") == old.get("value"), f"{label} support flags differ")
        rows.append(
            {
                "case_id": case["case_id"],
                "phases": phases,
                "registered_combined": combined,
                "combined": combined,
                "full": phases["ported"],
                "passed": len(mismatches) == start,
            }
        )
    return rows


def compare_metrics(rows, task, mismatches):
    result = {}
    try:
        reference = counts([row["original"]["value"] for row in rows])
        direct = counts([row["direct_ported"]["value"] for row in rows])
        registered = counts([row["registered"]["value"] for row in rows])
        aggregations = task.aggregation()
        for name, expected in reference.items():
            native = float(aggregations[name]([row["registered"]["value"][name] for row in rows]))
            error = abs(native - expected["value"])
            passed = expected == direct[name] == registered[name] and error <= 1e-12
            result[name] = {
                "reference": expected,
                "direct_ported": direct[name],
                "registered": registered[name],
                "registered_aggregation_value": native,
                "absolute_error": error,
                "passed": passed,
            }
            if not passed:
                mismatches.append({"metric": name, **result[name]})
    except (KeyError, TypeError, ValueError) as error:
        # Invalid scorer output must still produce the mismatch report.
        mismatches.append(
            {"source": "metric aggregation", "error": {"type": type(error).__name__, "message": str(error)}}
        )
    return result


def validate_dataset(task, source_root):
    import pyarrow.parquet as pq
    from datasets import Dataset, DatasetDict

    path = source_root / "sources/ko/data/train-00000-of-00001.parquet"
    rows = pq.read_table(path).to_pylist()
    task.dataset = DatasetDict({"train": Dataset.from_list(rows)})
    docs = task.eval_docs()
    require(len(rows) == len(docs) == len({row["key"] for row in rows}) == 342, "Wrong dataset document count")
    require(sum(len(row["instruction_id_list"]) for row in rows) == 481, "Wrong dataset instruction count")
    for row, doc in zip(rows, docs):
        require(
            doc.query == row["prompt"] and doc.specific["ifeval_ko"]["key"] == row["key"], "Dataset prompt/key changed"
        )
        require(
            doc.specific["instructions_id_list"] == row["instruction_id_list"]
            and doc.specific["kwargs"] == row["kwargs"],
            "Dataset instructions changed",
        )
    return {
        "status": "PASS",
        "prompts": len(docs),
        "instructions": 481,
        "source_sha256": sha(path),
        "model_scoring_run": False,
    }


def git_head():
    result = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def decision_summary(rows, cases):
    result = {mode: {"instructions": 0, "prompts": 0} for mode in KINDS}
    for row, case in zip(rows, cases):
        actual = row["registered"].get("value")
        if not isinstance(actual, dict):
            continue
        expected = oracle(case)
        for mode in KINDS:
            flags = actual.get(f"inst_level_{mode}_acc")
            if isinstance(flags, list):
                result[mode]["instructions"] += sum(
                    type(flag) is bool and flag == old for flag, old in zip(flags, expected[f"inst_level_{mode}_acc"])
                )
            flag = actual.get(f"prompt_level_{mode}_acc")
            result[mode]["prompts"] += type(flag) is bool and flag == expected[f"prompt_level_{mode}_acc"]
    return result


def run(args, report):
    reference, case_utils, evidence_hashes, sources = load_reference(args.audit_dir, args.source_root)
    task, scoring, response_cls, code_hashes = load_task()
    cases, support, stored, _ = load_cases(args.audit_dir)
    external = {
        "saved direct scorer": load_external(args.direct_results, "direct scorer", cases, support),
        "upstream registered task": load_external(args.upstream_results, "registered task", cases, support),
    }
    mismatches = report["mismatches"]
    rows = validate_normal(cases, stored, task, scoring, response_cls, reference, case_utils, external, mismatches)
    auxiliary = validate_support(support, task, scoring, response_cls, reference, case_utils, external, mismatches)
    report.update(results=rows, support_results=auxiliary)
    metrics = compare_metrics(rows, task, mismatches)
    dataset = validate_dataset(task, args.source_root)
    report.update(
        normal_cases=len(rows),
        matched_cases=sum(row["passed"] for row in rows),
        normal_scoring_errors=sum(
            "error" in row[engine] for row in rows for engine in ("original", "direct_ported", "registered")
        ),
        matched_decisions=decision_summary(rows, cases),
        support={
            "cases": len(auxiliary),
            "matched_cases": sum(row["passed"] for row in auxiliary),
            "normal_returns_per_mode": {
                mode: sum("error" not in row["phases"]["ported"][mode] for row in auxiliary) for mode in KINDS
            },
            "expected_exceptions_per_mode": {
                mode: sum("error" in row["phases"]["ported"][mode] for row in auxiliary) for mode in KINDS
            },
        },
        metrics=metrics,
        dataset_formatting=dataset,
        external_comparisons={
            label: {
                "status": "CHECKED" if value is not None else "NOT_RUN",
                "sha256": value["sha256"] if value else None,
            }
            for label, value in external.items()
        },
        environment={
            "python": sys.version,
            "platform": platform.platform(),
            "repo": str(REPO),
            "target_sha": git_head(),
            "audit_revision": AUDIT_REVISION,
            "ko_revision": KO_REVISION,
            "random_seed": 0,
            "langdetect_seed": 0,
            "workers": 1,
            "reference_import_mode": "isolated-source",
            "checker_mode": "ported",
            "code_sha256": code_hashes,
            "script_sha256": sha(Path(__file__)),
            "evidence_sha256": evidence_hashes,
            "verified_sources": sources,
            "package_versions": {
                name: importlib.metadata.version(name)
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
                )
            },
        },
    )
    if mismatches and "case_id" in mismatches[0]:
        first = copy.deepcopy(mismatches[0])
        case = next(case for case in cases + support if case["case_id"] == first["case_id"])
        first["input"] = {**case_utils.ko_input(case), "response": case["test_response_ko"]}
        report["first_mismatch"] = first
    report.update(passed=not mismatches, status="FAIL" if mismatches else "PASS")
    return int(bool(mismatches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--direct-results", type=Path, help="Optional saved --checker-mode ported output")
    parser.add_argument(
        "--upstream-results", type=Path, help="Optional registered-task attachment with per-case results"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new filename to preserve previous results")
    args.audit_dir, args.source_root = args.audit_dir.resolve(), args.source_root.resolve()
    sys.dont_write_bytecode = True
    report = {
        "status": "RUNNING",
        "passed": False,
        "command": [sys.executable, *sys.argv],
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Registry -> Doc -> ModelResponse -> compute_sample -> corpus aggregation; no model inference",
        "mismatches": [],
        "first_mismatch": None,
    }
    try:
        result = run(args, report)
    except Exception as error:
        report.update(
            status="ERROR",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        result = 2
    if report["mismatches"] and report["first_mismatch"] is None:
        first = copy.deepcopy(report["mismatches"][0])
        if "case_id" in first:
            for filename in ("cases.json", "support_cases.json"):
                for case in read(args.audit_dir / filename):
                    if case["case_id"] == first["case_id"]:
                        first["input"] = {
                            "key": case["key"],
                            "prompt": case["ko_prompt"],
                            "instruction_id_list": case["ko_instruction_id_list"],
                            "kwargs": case.get("ko_eval_all_kwargs", case["ko_all_kwargs"]),
                            "response": case["test_response_ko"],
                        }
        report["first_mismatch"] = first
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"environment", "results", "support_results"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Report: {args.output}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

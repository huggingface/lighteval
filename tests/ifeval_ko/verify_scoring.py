"""Compare the actual Ko scorer with pinned, read-only PR #30 evidence.

This verifies the scorer directly, not a registered Lighteval task or a model.
When checker ports are absent, --checker-mode pinned-source explicitly binds
only the validation namespace to the unmodified, hash-verified Ko registry.
"""

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import types
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "src/lighteval/tasks/tasks/ifeval_ko"
AUDIT_REVISION = "da7053db1d4858239856695a2cfed702173234b6"
KO_REVISION = "54199e3801116897697babf341865741dcd06fc8"
KINDS = ("strict", "loose")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_records(records, count):
    indexed = {row["case_id"]: row for row in records}
    require(len(records) == len(indexed) == count, f"Missing or duplicate case IDs; expected {count}")
    return indexed


def load_scorer(registry=None):
    """Load the real source without importing unrelated framework dependencies."""
    name = "_ifeval_ko_validation"
    package = types.ModuleType(name)
    package.__path__ = [str(SCORER_DIR)]
    sys.modules[name] = package
    if registry is not None:
        sys.modules[name + ".instructions_registry"] = registry
        package.instructions_registry = registry
    return importlib.import_module(name + ".scoring")


class Trace:
    """Observe candidate, checker lifetime, construction and call order in situ."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.phase_codes = {getattr(scorer, "score_" + kind).__code__: kind for kind in KINDS}
        self.objects = {}  # Hold references so object IDs cannot be recycled.
        self.inputs = {}
        self.schedule = []
        self.phase = None
        self.full = {
            kind: {
                field: []
                for field in (
                    "checker_call_trace",
                    "checker_build_trace",
                    "checker_creation_trace",
                    "loose_candidates_from_evaluator",
                )
            }
            for kind in KINDS
        }

    def profile(self, frame, event, arg):
        kind = self.phase_codes.get(frame.f_code)
        if kind is not None:
            self.record_phase(frame, event, arg, kind)
            return
        parent = frame.f_back
        if parent is None or self.phase is None:
            return
        if parent.f_code not in {*self.phase_codes, self.scorer._build_instruction.__code__}:
            return
        obj = frame.f_locals.get("self")
        method = frame.f_code.co_name
        if obj is None or not hasattr(obj, "id") or method not in {"__init__", "build_description", "check_following"}:
            return
        if event == "call" and method == "check_following":
            self.inputs[id(frame)] = frame.f_locals.get("value")
        if event != "return":
            return
        self.record_checker(frame, arg, obj, method, parent)

    def record_phase(self, frame, event, arg, kind):
        if event == "call":
            self.phase = kind
            self.schedule.append(kind)
        elif event == "return":
            out = self.full[kind]
            if "all_responses" in frame.f_locals:
                out["loose_candidates_from_evaluator"] = frame.f_locals["all_responses"].copy()
            if arg is not None:
                out.update(
                    instruction_results=arg.follow_instruction_list,
                    prompt_result=arg.follow_all_instructions,
                    evaluated_response=frame.f_locals["response"],
                )
            self.phase = None

    def record_checker(self, frame, arg, obj, method, parent):
        index = parent.f_locals["index"]
        if id(obj) not in self.objects:
            self.objects[id(obj)] = (obj, f"full.{self.phase}:{index}")
        base = {"instruction_id": obj.id, "instruction_index": index, "instance": self.objects[id(obj)][1]}
        out = self.full[self.phase]
        if method == "__init__":
            out["checker_creation_trace"].append(base)
        elif method == "build_description":
            try:
                built = copy.deepcopy(obj.get_instruction_args())
            except Exception as error:
                built = {"get_args_error": type(error).__name__}
            out["checker_build_trace"].append(
                {
                    **base,
                    "filtered_kwargs": copy.deepcopy(parent.f_locals["kwargs"]),
                    "built_args": built,
                }
            )
        else:
            out["checker_call_trace"].append({**base, "value": self.inputs.pop(id(frame)), "result": bool(arg)})

    def run(self, fn, *args):
        previous = sys.getprofile()
        try:
            sys.setprofile(self.profile)
            return {"value": fn(*args)}
        except Exception as error:
            return {"error": {"type": type(error).__name__, "message": str(error)}}
        finally:
            sys.setprofile(previous)


def trace_view(full):
    """Ignore diagnostics from checker-local variables; retain actual calls."""
    result = copy.deepcopy(full)
    for out in result.values():
        out.pop("error", None)
        for call in out["checker_call_trace"]:
            call.pop("checker_local_value", None)
            call.pop("counts", None)
    return result


def from_reference(record):
    return {
        key: record["full"][kind][field]
        for kind in KINDS
        for key, field in (
            (f"prompt_level_{kind}_acc", "prompt_result"),
            (f"inst_level_{kind}_acc", "instruction_results"),
        )
    }


def counts(values):
    result = {}
    for kind in KINDS:
        for level in ("prompt", "inst"):
            key = f"{level}_level_{kind}_acc"
            flags = [v[key] for v in values] if level == "prompt" else [f for v in values for f in v[key]]
            require(flags and all(type(f) is bool for f in flags), f"Invalid Boolean denominator for {key}")
            result[key] = {"numerator": sum(flags), "denominator": len(flags), "value": sum(flags) / len(flags)}
    return result


def compare_flags(case, actual, reference, mismatches, source):
    ids = case["ko_instruction_id_list"]
    for kind in KINDS:
        expected = reference[f"inst_level_{kind}_acc"]
        flags = actual.get(f"inst_level_{kind}_acc", [])
        for i in range(max(len(ids), len(flags))):
            old = expected[i] if i < len(expected) else None
            new = flags[i] if i < len(flags) else None
            if type(new) is not bool or old != new:
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "key": case["key"],
                        "mode": kind,
                        "instruction_index": i,
                        "instruction_id": ids[i] if i < len(ids) else None,
                        "source": source,
                        "reference": old,
                        "new": new,
                    }
                )
        old, new = reference[f"prompt_level_{kind}_acc"], actual.get(f"prompt_level_{kind}_acc")
        if type(new) is not bool or old != new:
            mismatches.append(
                {
                    "case_id": case["case_id"],
                    "key": case["key"],
                    "mode": kind,
                    "instruction_index": None,
                    "instruction_id": None,
                    "source": source,
                    "reference": old,
                    "new": new,
                }
            )


def oracle(case):
    expected = case["expected_ko"]
    return {
        f"{level}_level_{kind}_acc": expected[field][kind]
        for kind in KINDS
        for level, field in (("inst", "full"), ("prompt", "prompt"))
    }


def load_environment(args):
    audit, source_root = args.audit_dir.resolve(), args.source_root.resolve()
    sys.dont_write_bytecode = True
    # Verify all evidence bytes before importing its validation helpers.
    hashes = {}
    for line in (audit / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        require(sha(audit / filename) == digest, f"Evidence hash mismatch: {filename}")
        hashes[filename] = digest
    os.environ["NLTK_DATA"] = str(source_root / ".cache/nltk_data")
    os.environ["IFEVAL_AUDIT_SOURCE_ROOT"] = str(source_root)
    sys.path.insert(0, str(audit))
    import audit_runtime
    import case_utils
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    audit_runtime.SOURCE_ROOT = source_root
    provenance, verified = case_utils.verify_sources(source_root)
    require(provenance["ko"]["revision"] == KO_REVISION, "Wrong Ko revision")
    nltk_hashes = {}
    for filename, digest in read(audit / "nltk_resources.json")["sha256"].items():
        require(sha(source_root / ".cache/nltk_data" / filename) == digest, f"NLTK resource mismatch: {filename}")
        nltk_hashes[filename] = digest
    versions = {
        name: importlib.metadata.version(name)
        for name in ("nltk", "langdetect", "immutabledict", "packaging", "pyarrow")
    }
    require(
        versions == read(audit / "ko_results.json")["environment"]["package_versions"],
        "Dependency versions differ from the stored Ko run",
    )
    # Explicit source-only reference environment, with original imports intact.
    modules = audit_runtime.load_modules("ko", "isolated-source")
    for name, module in modules.items():
        require(
            sha(Path(module.__file__)) == provenance["ko"]["files"][f"ifeval_ko/{name}.py"]["sha256"],
            f"Imported reference hash mismatch: {name}",
        )
    scorer = load_scorer(modules["instructions_registry"] if args.checker_mode == "pinned-source" else None)
    return modules, scorer, verified, hashes, nltk_hashes, versions


def load_cases(audit):
    import case_utils

    cases = read(audit / "cases.json")
    support = read(audit / "support_cases.json")
    case_utils.validate_cases(cases)
    case_utils.validate_cases(support, require_expected=False)
    source_rows = {row["key"]: row for row in read(audit / "source_rows.json")}
    for case in cases:
        case_utils.validate_source_row(case, source_rows)
    index_records(cases, 156)
    index_records(support, 18)
    stored = index_records(read(audit / "ko_results.json")["results"], 156)
    stored_support = index_records(read(audit / "support_ko_results.json")["results"], 18)
    require(set(stored) == {c["case_id"] for c in cases}, "Reference case IDs differ")
    require(set(stored_support) == {c["case_id"] for c in support}, "Reference support IDs differ")
    require(sum(len(c["ko_instruction_id_list"]) for c in cases) == 247, "Wrong instruction count")
    return cases, support, stored, stored_support


def evaluate_normal(cases, stored, scorer, modules, mismatches):
    import case_utils
    import compare

    rows = []
    for case in cases:
        record = stored[case["case_id"]]
        compare.verify_record(case, record, "ko")
        raw = case_utils.ko_input(case)
        inputs = (raw["prompt"], raw["instruction_id_list"], raw["kwargs"], case["test_response_ko"])
        original_input = copy.deepcopy(inputs)
        mismatch_count = len(mismatches)
        random.seed(0)
        tracer = Trace(scorer)
        actual = tracer.run(scorer.score_response, *inputs)
        require(inputs == original_input, f"Scorer modified input: {case['case_id']}")
        if "error" in actual:
            mismatches.append({"case_id": case["case_id"], "key": case["key"], "error": actual["error"]})
        else:
            compare_flags(case, actual["value"], oracle(case), mismatches, "predeclared oracle")
            compare_flags(case, actual["value"], from_reference(record), mismatches, "stored Ko")
            random.seed(0)
            live = modules["utils"].process_results(raw, [inputs[-1]])
            compare_flags(case, actual["value"], live, mismatches, "live pinned Ko")
            if tracer.schedule != list(KINDS) or trace_view(tracer.full) != trace_view(record["full"]):
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "key": case["key"],
                        "source": "call trace",
                        "schedule": tracer.schedule,
                        "reference": trace_view(record["full"]),
                        "new": trace_view(tracer.full),
                    }
                )
        rows.append(
            {
                "case_id": case["case_id"],
                "key": case["key"],
                "instruction_ids": inputs[1],
                "input_sha256": case["hashes"]["input_sha256"],
                "response_sha256": case["hashes"]["response_sha256"],
                "passed": len(mismatches) == mismatch_count,
                **actual,
            }
        )
    return rows


def evaluate_support(support, stored_support, scorer, mismatches):
    import case_utils
    import compare

    support_rows = []
    for case in support:
        record = stored_support[case["case_id"]]
        compare.verify_support_record(case, record, "ko")
        raw = case_utils.ko_input(case)
        inputs = (raw["prompt"], raw["instruction_id_list"], raw["kwargs"], case["test_response_ko"])
        random.seed(0)
        phases, tracer = {}, Trace(scorer)
        # Evaluate both entries even when strict raises; do not count exceptions
        # as successful normal scores or add support cases to metric denominators.
        for kind in KINDS:
            phases[kind] = tracer.run(getattr(scorer, "score_" + kind), *inputs)
            expected_error = case["expected_ko"].get("error_type")
            found_error = phases[kind].get("error", {}).get("type")
            if expected_error != found_error:
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "key": case["key"],
                        "mode": kind,
                        "reference_error": expected_error,
                        "new": phases[kind].get("error"),
                    }
                )
            if "value" in phases[kind]:
                value = phases[kind].pop("value")
                phases[kind].update(
                    instruction_results=value.follow_instruction_list, prompt_result=value.follow_all_instructions
                )
                if not expected_error and (
                    phases[kind]["instruction_results"] != case["expected_ko"]["full"][kind]
                    or phases[kind]["prompt_result"] != case["expected_ko"]["prompt"][kind]
                ):
                    mismatches.append(
                        {
                            "case_id": case["case_id"],
                            "key": case["key"],
                            "mode": kind,
                            "reference": case["expected_ko"],
                            "new": phases[kind],
                        }
                    )
        if trace_view(tracer.full) != trace_view(record["full"]):
            mismatches.append(
                {
                    "case_id": case["case_id"],
                    "key": case["key"],
                    "source": "support call trace",
                    "reference": trace_view(record["full"]),
                    "new": trace_view(tracer.full),
                }
            )
        # Confirm the public combined entry also propagates strict exceptions.
        random.seed(0)
        combined = Trace(scorer).run(scorer.score_response, *inputs)
        if case["expected_ko"].get("error_type"):
            if combined.get("error", {}).get("type") != case["expected_ko"]["error_type"]:
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "key": case["key"],
                        "instruction_index": case["instruction_index"],
                        "instruction_id": case["instruction_id"],
                        "source": "support combined exception",
                        "reference": case["expected_ko"],
                        "new": combined,
                    }
                )
        else:
            compare_flags(case, combined.get("value", {}), oracle(case), mismatches, "support combined")
        support_rows.append({"case_id": case["case_id"], "full": phases, "combined": combined})
    return support_rows


def compare_metrics(rows, cases, stored, mismatches):
    metric_comparison = {}
    if all("value" in row for row in rows):
        new_counts = counts([row["value"] for row in rows])
        old_counts = counts([from_reference(stored[c["case_id"]]) for c in cases])
        for key, old in old_counts.items():
            new = new_counts[key]
            difference = abs(old["value"] - new["value"])
            passed = old == new and difference <= 1e-12
            metric_comparison[key] = {"reference": old, "new": new, "absolute_error": difference, "passed": passed}
            if not passed:
                mismatches.append({"metric": key, **metric_comparison[key]})
    return metric_comparison


def support_counts(rows):
    return {
        "cases": len(rows),
        "normal_returns_per_mode": {kind: sum("error" not in row["full"][kind] for row in rows) for kind in KINDS},
        "exceptions_per_mode": {kind: sum("error" in row["full"][kind] for row in rows) for kind in KINDS},
    }


def decision_matches(rows, cases):
    by_id = {case["case_id"]: oracle(case) for case in cases}
    matched = {}
    for kind in KINDS:
        key = f"inst_level_{kind}_acc"
        matched[kind] = {
            "instructions": sum(
                type(flag) is bool and flag == old
                for row in rows
                for flag, old in zip(row.get("value", {}).get(key, []), by_id[row["case_id"]][key])
            ),
            "prompts": sum(
                type(row.get("value", {}).get(f"prompt_level_{kind}_acc")) is bool
                and row["value"][f"prompt_level_{kind}_acc"] == by_id[row["case_id"]][f"prompt_level_{kind}_acc"]
                for row in rows
            ),
        }
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checker-mode", required=True, choices=("pinned-source", "ported"))
    parser.add_argument("--output", type=Path, default=REPO / "results/ifeval_ko/scoring-parity.json")
    args = parser.parse_args()
    audit, source_root = args.audit_dir.resolve(), args.source_root.resolve()
    modules, scorer, verified, hashes, nltk_hashes, versions = load_environment(args)
    cases, support, stored, stored_support = load_cases(audit)
    mismatches = []
    rows = evaluate_normal(cases, stored, scorer, modules, mismatches)
    support_rows = evaluate_support(support, stored_support, scorer, mismatches)
    metric_comparison = compare_metrics(rows, cases, stored, mismatches)
    first = copy.deepcopy(mismatches[0]) if mismatches else None
    if first and "case_id" in first:
        import case_utils

        case = next(c for c in cases + support if c["case_id"] == first["case_id"])
        first["input"] = {**case_utils.ko_input(case), "response": case["test_response_ko"]}
    report = {
        "passed": not mismatches,
        "scope": "direct scorer; registered task path not exercised",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "command": [sys.executable, *sys.argv],
            "audit_revision": AUDIT_REVISION,
            "ko_revision": KO_REVISION,
            "random_seed": 0,
            "langdetect_seed": 0,
            "workers": 1,
            "checker_mode": args.checker_mode,
            "reference_import_mode": "isolated-source",
            "audit_dir": str(audit),
            "source_root": str(source_root),
            "scorer_sha256": sha(SCORER_DIR / "scoring.py"),
            "verifier_sha256": sha(Path(__file__)),
            "evidence_sha256": hashes,
            "verified_sources": verified,
            "package_versions": versions,
            "nltk_resource_sha256": nltk_hashes,
        },
        "normal_cases": len(rows),
        "matched_cases": sum(row["passed"] for row in rows),
        "matched_decisions": decision_matches(rows, cases),
        "instructions_per_mode": sum(len(row["instruction_ids"]) for row in rows),
        "prompts_per_mode": len(rows),
        "normal_scoring_errors": sum("error" in row for row in rows),
        "support": support_counts(support_rows),
        "metrics": metric_comparison,
        "mismatches": mismatches,
        "first_mismatch": first,
        "results": rows,
        "support_results": support_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "passed",
                    "normal_cases",
                    "instructions_per_mode",
                    "normal_scoring_errors",
                    "support",
                    "metrics",
                )
            },
            indent=2,
        )
    )
    print(f"Report: {args.output}")
    if first:
        print(json.dumps(first, ensure_ascii=False, indent=2))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())

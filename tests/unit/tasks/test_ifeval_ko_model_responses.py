"""Negative controls for saved-response integrity, comparison and aggregation."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch


VERIFIER = Path(__file__).resolve().parents[2] / "ifeval_ko/verify_model_responses.py"


class Response:
    def __init__(self, text):
        self.text = text

    @property
    def final_text(self):
        return self.text


class Document:
    __slots__ = ("query", "choices", "gold_index", "instruction", "specific")

    def __init__(self, raw):
        self.query = raw["prompt"]
        self.choices = [""]
        self.gold_index = 0
        self.instruction = ""
        self.specific = {
            "instructions_id_list": raw["instruction_id_list"],
            "kwargs": raw["kwargs"],
            "ifeval_ko": {"key": raw["key"]},
        }


class ModelResponseTests(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, "path", [str(VERIFIER.parent), *sys.path]):
            spec = importlib.util.spec_from_file_location("_test_model_response_verifier", VERIFIER)
            self.verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.verifier)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.detector_factory = types.SimpleNamespace(seed=0)
        language_module = types.SimpleNamespace(DetectorFactory=self.detector_factory)
        modules = patch.dict(sys.modules, {"langdetect": language_module})
        modules.start()
        self.addCleanup(modules.stop)
        self.next_run = 0
        self.keys = self.verifier.agreed_keys()
        self.source_rows = []
        offset = 0
        for index, key in enumerate(self.keys + list(range(100000, 100319))):
            size = 2 if index < 17 or 23 <= index < 145 else 1
            ids = [f"type-{i % 21}" for i in range(offset, offset + size)]
            if key == 1203:
                ids[1] = ids[0]
            self.source_rows.append(
                {
                    "key": key,
                    "prompt": f"Prompt {key}",
                    "instruction_id_list": ids,
                    "kwargs": [{"limit": 1} for _ in range(size)],
                }
            )
            offset += size

    def write_json(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def write_records(self, directory, records):
        (directory / "responses.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )

    def refresh_hashes(self, directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["artifact_sha256"] = {
            name: self.verifier.sha(directory / name) for name in self.verifier.RUN_FILES if name != "manifest.json"
        }
        self.write_json(directory / "manifest.json", manifest)
        (directory / "SHA256SUMS").write_text(
            "".join(f"{self.verifier.sha(directory / name)}  {name}\n" for name in sorted(self.verifier.RUN_FILES)),
            encoding="utf-8",
        )

    def make_run(self, scope="selected"):
        self.next_run += 1
        directory = self.root / f"input-{self.next_run}"
        directory.mkdir()
        rows = copy.deepcopy(self.source_rows if scope == "full" else self.source_rows[:23])
        keys = [row["key"] for row in rows]
        records = []
        for row in rows:
            response = f"\n  \u1100\u1161 response {row['key']} \n"
            length = row["key"] == 1127
            tokens = [2] * (1280 if length else 1)
            records.append(
                {
                    **copy.deepcopy(row),
                    "status": "generated",
                    "response": response,
                    "scoring_response": response,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "finish_reason": "length" if length else "eos",
                    "output_token_ids": tokens,
                    "generated_token_count": len(tokens),
                }
            )
        self.write_records(directory, records)
        self.write_json(directory / "selected-samples.json", rows)
        self.write_json(directory / "selection.json", {"keys": keys, "scope": scope})
        self.write_json(
            directory / "manifest.json",
            {
                "status": "responses_generated",
                "evaluation_scope": scope,
                "expected_count": len(rows),
                "generated_count": len(rows),
                "generation_error_count": 0,
                "unattempted_count": 0,
                "instruction_occurrences": sum(len(row["instruction_id_list"]) for row in rows),
                "dataset_id": "allganize/IFEval-Ko",
                "dataset_revision": self.verifier.KO_REVISION,
                "dataset_subset": "default",
                "dataset_split": "train",
                "num_fewshots": 0,
                "selected_keys": keys,
                "empty_response_count": 0,
                "length_limited_count": 1,
                "code_revision": "generation-revision",
            },
        )
        (directory / "generate_responses.py").write_text("# original generation snapshot, not current code\n")
        (directory / "run.log").write_text("original log\n")
        (directory / "requirements.lock.txt").write_text("original environment\n")
        self.refresh_hashes(directory)
        return directory, records

    def make_archive(self, directory):
        archive = self.root / f"{directory.name}.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name in self.verifier.RUN_FILES | {"SHA256SUMS"}:
                bundle.write(directory / name, f"saved-run/{name}")
            bundle.writestr("saved-run/._generate_responses.py", "unscored resource fork")
        return archive

    def decisions(self, raw):
        strict = [True] + [False] * (len(raw["instruction_id_list"]) - 1)
        return {
            "inst_level_strict_acc": strict,
            "prompt_level_strict_acc": all(strict),
            "inst_level_loose_acc": [True] * len(strict),
            "prompt_level_loose_acc": True,
        }

    def engines(self):
        seen = []

        def score(engine, raw, text):
            seen.append((engine, copy.deepcopy(raw), text))
            return self.decisions(raw)

        def formatter(raw, name):
            return Document(raw)

        def compute(doc, model_response):
            raw = {
                "key": doc.specific["ifeval_ko"]["key"],
                "prompt": doc.query,
                "instruction_id_list": doc.specific["instructions_id_list"],
                "kwargs": doc.specific["kwargs"],
            }
            return score("registered", raw, model_response.final_text[0])

        def mean(items):
            return sum(items) / len(items)

        def flat_mean(items):
            flags = [flag for group in items for flag in group]
            return mean(flags)

        aggregation = {name: mean if name.startswith("prompt") else flat_mean for name in self.verifier.METRICS}
        task = types.SimpleNamespace(
            name="ifeval_ko|0",
            formatter=formatter,
            metrics=[types.SimpleNamespace(compute_sample=compute)],
            aggregation=Mock(return_value=aggregation),
        )
        reference = types.SimpleNamespace(process_results=lambda raw, texts: score("original", raw, texts[0]))

        def direct(prompt, ids, kwargs, text):
            raw = {"prompt": prompt, "instruction_id_list": ids, "kwargs": kwargs}
            return score("direct_ported", raw, text)

        return reference, task, types.SimpleNamespace(score_response=direct), Response, seen

    def test_archive_and_inputs_are_preserved_with_old_generator_snapshot(self):
        directory, records = self.make_run()
        archive = self.make_archive(directory)
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        result = self.verifier.load_run(directory, archive, self.verifier.sha(archive))
        self.verifier.check_source_rows(result["records"], self.source_rows)
        self.assertEqual(result["records"], records)
        self.assertEqual(result["integrity"]["archive"]["status"], "PASS")
        self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})
        self.assertEqual(next(row for row in records if row["key"] == 1127)["finish_reason"], "length")

    def test_archive_hash_and_payload_must_match(self):
        directory, _ = self.make_run()
        archive = self.make_archive(directory)
        with self.assertRaisesRegex(ValueError, "Archive SHA-256"):
            self.verifier.load_run(directory, archive, "0" * 64)
        (directory / "run.log").write_text("changed but internally rehashed\n")
        self.refresh_hashes(directory)
        with self.assertRaisesRegex(ValueError, "Archive/run mismatch"):
            self.verifier.load_run(directory, archive, self.verifier.sha(archive))

    def test_unicode_line_separators_in_responses_are_preserved(self):
        for separator in ("\u2028", "\u2029", "\u0085"):
            with self.subTest(separator=f"U+{ord(separator):04X}"):
                directory, records = self.make_run()
                response = records[0]["response"] + f"before{separator}after"
                records[0].update(
                    response=response,
                    scoring_response=response,
                    response_sha256=hashlib.sha256(response.encode("utf-8")).hexdigest(),
                )
                self.write_records(directory, records)
                self.refresh_hashes(directory)
                archive = self.make_archive(directory)
                result = self.verifier.load_run(directory, archive, self.verifier.sha(archive))
                self.assertEqual(result["records"], records)
                self.verifier.check_source_rows(result["records"], self.source_rows)

    def test_checksum_and_response_hash_tampering_are_rejected(self):
        for rehash in (False, True):
            with self.subTest(rehash=rehash):
                directory, records = self.make_run()
                records[0]["response"] += "changed"
                records[0]["scoring_response"] = records[0]["response"]
                self.write_records(directory, records)
                if rehash:
                    self.refresh_hashes(directory)
                with self.assertRaisesRegex(ValueError, "Response hash|Run checksum"):
                    self.verifier.load_run(directory)

    def test_missing_duplicate_and_reordered_records_are_rejected(self):
        for change in ("missing", "duplicate", "reordered"):
            with self.subTest(change=change):
                directory, records = self.make_run()
                if change == "missing":
                    records.pop()
                elif change == "duplicate":
                    records[-1] = records[0]
                else:
                    records.reverse()
                self.write_records(directory, records)
                self.refresh_hashes(directory)
                with self.assertRaisesRegex(ValueError, "Missing or extra|reordered keys"):
                    self.verifier.load_run(directory)

    def test_self_consistent_kwargs_tampering_is_rejected_against_source(self):
        directory, records = self.make_run()
        selected = json.loads((directory / "selected-samples.json").read_text())
        # True == 1 in Python, so the source comparison must also preserve types.
        records[0]["kwargs"][0]["limit"] = True
        selected[0]["kwargs"][0]["limit"] = True
        self.write_records(directory, records)
        self.write_json(directory / "selected-samples.json", selected)
        self.refresh_hashes(directory)
        saved = self.verifier.load_run(directory)
        with self.assertRaisesRegex(ValueError, "Pinned source input differs"):
            self.verifier.check_source_rows(saved["records"], self.source_rows)

    def test_incomplete_generation_cannot_be_padded_as_success(self):
        directory, records = self.make_run()
        records[-1]["status"] = "error"
        self.write_records(directory, records)
        self.refresh_hashes(directory)
        with self.assertRaisesRegex(ValueError, "not successfully generated"):
            self.verifier.load_run(directory)

    def test_complete_run_with_only_logging_error_remains_scorable(self):
        directory, _ = self.make_run()
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest.update(status="error", error_stage="logging", logging_errors=[{"type": "BrokenPipeError"}])
        self.write_json(directory / "manifest.json", manifest)
        self.refresh_hashes(directory)
        self.assertEqual(len(self.verifier.load_run(directory)["records"]), 23)

    def test_empty_generated_response_is_preserved(self):
        directory, records = self.make_run()
        records[0].update(response="", scoring_response="", response_sha256=hashlib.sha256(b"").hexdigest())
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["empty_response_count"] = 1
        self.write_json(directory / "manifest.json", manifest)
        self.write_records(directory, records)
        self.refresh_hashes(directory)
        self.assertEqual(self.verifier.load_run(directory)["records"][0]["response"], "")

    def test_three_paths_keep_exact_text_and_instruction_weighted_denominators(self):
        _, records = self.make_run()
        reference, task, scoring, response_class, seen = self.engines()
        report = self.verifier.score_records(records, reference, task, scoring, response_class)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["matched_decisions"], {mode: {"prompts": 23, "instructions": 40} for mode in ("strict", "loose")}
        )
        self.assertEqual(report["metrics"]["inst_level_strict_acc"]["original"]["value"], 23 / 40)
        self.assertEqual(report["metrics"]["prompt_level_strict_acc"]["original"]["value"], 6 / 23)
        self.assertEqual([text for _, _, text in seen], [row["response"] for row in records for _ in range(3)])
        for row in report["results"]:
            for engine in self.verifier.ENGINES:
                execution = row[engine]["execution"]
                self.assertEqual(execution["input_sha256"], row["input_sha256"])
                self.assertEqual(execution["response_sha256"], row["response_sha256"])

    def test_random_and_language_seeds_are_reset_before_every_engine_call(self):
        _, records = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        original, direct, registered = (
            reference.process_results,
            scoring.score_response,
            task.metrics[0].compute_sample,
        )
        observed = []

        def observe(fn, *args):
            observed.append((random.random(), self.detector_factory.seed))
            result = fn(*args)
            random.seed(999)
            self.detector_factory.seed = 999
            return result

        reference.process_results = lambda raw, texts: observe(original, raw, texts)
        scoring.score_response = lambda prompt, ids, kwargs, text: observe(direct, prompt, ids, kwargs, text)
        task.metrics[0].compute_sample = lambda doc, model_response: observe(registered, doc, model_response)
        report = self.verifier.score_records(records, reference, task, scoring, response_class)
        self.assertTrue(report["passed"])
        self.assertEqual(len(observed), 69)
        self.assertTrue(all(item == (observed[0][0], 0) for item in observed))
        states = {
            row[engine]["execution"]["seeds"]["python_state_sha256"]
            for row in report["results"]
            for engine in self.verifier.ENGINES
        }
        self.assertEqual(len(states), 1)

    def test_repeated_id_second_occurrence_mismatch_is_identified(self):
        _, records = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        compute = task.metrics[0].compute_sample

        def wrong(doc, model_response):
            result = compute(doc, model_response)
            if doc.specific["ifeval_ko"]["key"] == 1203:
                result["inst_level_strict_acc"][1] = True
                result["prompt_level_strict_acc"] = True
            return result

        task.metrics[0].compute_sample = wrong
        report = self.verifier.score_records(records, reference, task, scoring, response_class)
        self.assertFalse(report["passed"])
        mismatch = report["first_mismatch"]
        self.assertEqual((mismatch["key"], mismatch["instruction_index"]), (1203, 1))
        row = next(row for row in records if row["key"] == 1203)
        self.assertEqual(mismatch["instruction_id"], row["instruction_id_list"][1])
        self.assertEqual(mismatch["input"]["response"], row["response"])
        saved = next(row for row in report["results"] if row["key"] == 1203)
        entries = saved["instruction_comparisons"]
        self.assertEqual([(entry["key"], entry["instruction_index"]) for entry in entries], [(1203, 0), (1203, 1)])
        self.assertEqual(entries[0]["instruction_id"], entries[1]["instruction_id"])
        self.assertTrue(entries[0]["strict"]["matched"])
        self.assertFalse(entries[1]["strict"]["matched"])

    def test_native_prompt_averaging_of_instruction_scores_is_rejected(self):
        _, records = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        task.aggregation.return_value["inst_level_strict_acc"] = lambda groups: sum(
            sum(group) / len(group) for group in groups
        ) / len(groups)
        report = self.verifier.score_records(records, reference, task, scoring, response_class)
        self.assertFalse(report["passed"])
        self.assertFalse(report["metrics"]["inst_level_strict_acc"]["passed"])
        self.assertEqual(report["first_mismatch"]["input"]["scope"], "corpus")
        self.assertEqual(len(report["first_mismatch"]["input"]["records"]), 23)

    def test_equal_exceptions_fail_without_dropping_a_prompt_from_metrics(self):
        _, records = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        original, direct, registered = (
            reference.process_results,
            scoring.score_response,
            task.metrics[0].compute_sample,
        )
        target = records[0]

        def fail_first(condition, fn, *args):
            if condition:
                raise RuntimeError("same error in all paths")
            return fn(*args)

        reference.process_results = lambda raw, texts: fail_first(raw["key"] == target["key"], original, raw, texts)
        scoring.score_response = lambda prompt, ids, kwargs, text: fail_first(
            prompt == target["prompt"], direct, prompt, ids, kwargs, text
        )
        task.metrics[0].compute_sample = lambda doc, model_response: fail_first(
            doc.specific["ifeval_ko"]["key"] == target["key"], registered, doc, model_response
        )
        report = self.verifier.score_records(records, reference, task, scoring, response_class)
        self.assertFalse(report["passed"])
        self.assertEqual(report["scoring_errors"], 3)
        self.assertEqual(report["metrics_status"], "INCOMPLETE")
        self.assertEqual(report["metrics"], {})
        self.assertEqual(len(report["results"]), 23)
        task.aggregation.assert_not_called()

    def test_non_boolean_output_and_input_mutation_are_rejected(self):
        for mutation in ("boolean", "input"):
            with self.subTest(mutation=mutation):
                _, records = self.make_run()
                before = copy.deepcopy(records)
                reference, task, scoring, response_class, _ = self.engines()
                original = scoring.score_response

                def wrong(prompt, ids, kwargs, text):
                    result = original(prompt, ids, kwargs, text)
                    if mutation == "boolean":
                        result["inst_level_strict_acc"][0] = 1
                    else:
                        kwargs[0]["limit"] = 99
                    return result

                scoring.score_response = wrong
                report = self.verifier.score_records(records, reference, task, scoring, response_class)
                self.assertFalse(report["passed"])
                self.assertEqual(report["metrics_status"], "INCOMPLETE")
                self.assertEqual(records, before)

    def test_missing_or_extra_instruction_judgments_cannot_shrink_denominators(self):
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                _, records = self.make_run()
                reference, task, scoring, response_class, _ = self.engines()
                original = scoring.score_response

                def wrong(prompt, ids, kwargs, text):
                    result = original(prompt, ids, kwargs, text)
                    flags = result["inst_level_strict_acc"]
                    if change == "missing":
                        flags.pop()
                    else:
                        flags.append(True)
                    result["prompt_level_strict_acc"] = all(flags)
                    return result

                scoring.score_response = wrong
                report = self.verifier.score_records(records, reference, task, scoring, response_class)
                self.assertFalse(report["passed"])
                self.assertEqual(report["metrics"], {})
                self.assertEqual(report["metrics_status"], "INCOMPLETE")
                task.aggregation.assert_not_called()

    def test_registered_postprocessing_and_doc_mutation_are_rejected(self):
        class ChangedResponse(Response):
            @property
            def final_text(self):
                return [value.strip() for value in self.text]

        for change in ("response", "doc", "doc_input_types"):
            with self.subTest(change=change):
                _, records = self.make_run()
                before = copy.deepcopy(records)
                reference, task, scoring, response_class, _ = self.engines()
                compute = task.metrics[0].compute_sample

                def mutate_doc(doc, model_response):
                    result = compute(doc, model_response)
                    doc.specific["kwargs"][0]["limit"] = 99
                    return result

                if change == "doc_input_types":
                    formatter = task.formatter

                    def change_type(raw, name):
                        doc = formatter(raw, name)
                        doc.specific["kwargs"][0]["limit"] = True
                        return doc

                    task.formatter = change_type
                task.metrics[0].compute_sample = mutate_doc if change == "doc" else compute
                response_class = ChangedResponse if change == "response" else response_class
                report = self.verifier.score_records(records, reference, task, scoring, response_class)
                self.assertFalse(report["passed"])
                self.assertEqual(report["first_mismatch"]["source"], "registered")
                self.assertEqual(report["metrics_status"], "INCOMPLETE")
                self.assertEqual(records, before)

    def test_scoring_scope_requires_all_23_unique_prompts_and_40_instructions(self):
        for change in ("missing", "duplicate"):
            with self.subTest(change=change):
                _, records = self.make_run()
                reference, task, scoring, response_class, _ = self.engines()
                if change == "missing":
                    records.pop()
                else:
                    records[1] = records[0]
                with self.assertRaises(ValueError):
                    self.verifier.score_records(records, reference, task, scoring, response_class)

    def cli_args(self, directory, output, scope="selected"):
        return [
            "--scope",
            scope,
            "--run-dir",
            str(directory),
            "--audit-dir",
            str(self.root / "audit"),
            "--source-root",
            str(self.root / "sources"),
            "--output-dir",
            str(output),
        ]

    def run_scoring_cli(self, directory, output, reference, task, scoring, response_class, scope="selected"):
        source = self.root / "sources/sources/ko/data/train-00000-of-00001.parquet"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"unit-test dataset stand-in")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.read_table = lambda path: types.SimpleNamespace(to_pylist=lambda: copy.deepcopy(self.source_rows))
        arrow = types.ModuleType("pyarrow")
        arrow.__path__ = []
        arrow.parquet = parquet
        with (
            patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}),
            patch.object(self.verifier, "load_reference", return_value=(reference, None, {}, {})),
            patch.object(self.verifier, "load_task", return_value=(task, scoring, response_class, {})),
            patch.dict(os.environ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = self.verifier.main(self.cli_args(directory, output, scope))
        report = json.loads((output / "model-response-parity.json").read_text())
        for line in (output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            self.assertEqual(self.verifier.sha(output / name), digest)
        return code, report

    def test_full_split_archive_and_three_scoring_paths_use_342_and_481(self):
        directory, records = self.make_run(scope="full")
        archive = self.make_archive(directory)
        saved = self.verifier.load_run(directory, archive, self.verifier.sha(archive), scope="full")
        self.verifier.check_source_rows(saved["records"], self.source_rows, scope="full")
        reference, task, scoring, response_class, seen = self.engines()
        code, report = self.run_scoring_cli(
            directory, self.root / "full-parity", reference, task, scoring, response_class, scope="full"
        )
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["expected_prompts"], 342)
        self.assertEqual(report["expected_instructions"], 481)
        self.assertEqual(report["length_limited_keys"], [1127])
        self.assertEqual(
            report["matched_decisions"], {mode: {"prompts": 342, "instructions": 481} for mode in ("strict", "loose")}
        )
        self.assertEqual(report["metrics"]["inst_level_strict_acc"]["original"]["value"], 342 / 481)
        self.assertEqual(len(seen), 342 * 3)
        self.assertEqual([text for _, _, text in seen], [row["response"] for row in records for _ in range(3)])
        markdown = (self.root / "full-parity/model-response-parity.md").read_text()
        self.assertIn("342 prompts / 481 instructions", markdown)
        self.assertIn("342/342 prompt and 481/481 instruction", markdown)

    def test_full_split_cannot_be_used_as_selected_run_or_vice_versa(self):
        full, _ = self.make_run(scope="full")
        selected, _ = self.make_run()
        with self.assertRaises(ValueError):
            self.verifier.load_run(full)
        with self.assertRaises(ValueError):
            self.verifier.load_run(selected, scope="full")

    def test_full_split_rejects_self_consistent_reordering_against_pinned_source(self):
        directory, records = self.make_run(scope="full")
        records.reverse()
        keys = [row["key"] for row in records]
        self.write_records(directory, records)
        self.write_json(directory / "selected-samples.json", [self.verifier.raw_input(row) for row in records])
        self.write_json(directory / "selection.json", {"scope": "full", "keys": keys})
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["selected_keys"] = keys
        self.write_json(directory / "manifest.json", manifest)
        self.refresh_hashes(directory)
        saved = self.verifier.load_run(directory, scope="full")
        with self.assertRaisesRegex(ValueError, "key order"):
            self.verifier.check_source_rows(saved["records"], self.source_rows, scope="full")

    def test_full_split_missing_duplicate_and_changed_instruction_count_fail(self):
        for change in ("missing", "duplicate", "instruction"):
            with self.subTest(change=change):
                directory, records = self.make_run(scope="full")
                if change == "missing":
                    records.pop()
                elif change == "duplicate":
                    records[-1] = records[0]
                else:
                    records[0]["instruction_id_list"].pop()
                    records[0]["kwargs"].pop()
                self.write_records(directory, records)
                self.refresh_hashes(directory)
                with self.assertRaises(ValueError):
                    self.verifier.load_run(directory, scope="full")
                reference, task, scoring, response_class, _ = self.engines()
                with self.assertRaises(ValueError):
                    self.verifier.score_records(records, reference, task, scoring, response_class, scope="full")

    def test_full_split_scoring_failure_keeps_full_denominators_and_no_aggregate(self):
        directory, _ = self.make_run(scope="full")
        reference, task, scoring, response_class, _ = self.engines()
        original = reference.process_results

        def fail_one(raw, texts):
            if raw["key"] == self.source_rows[-1]["key"]:
                raise RuntimeError("full-split scoring failure")
            return original(raw, texts)

        reference.process_results = fail_one
        code, report = self.run_scoring_cli(
            directory, self.root / "full-error", reference, task, scoring, response_class, scope="full"
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(report["results"]), 342)
        self.assertEqual(report["expected_prompts"], 342)
        self.assertEqual(report["expected_instructions"], 481)
        self.assertEqual(report["scoring_errors"], 1)
        self.assertEqual(report["metrics_status"], "INCOMPLETE")
        self.assertEqual(report["metrics"], {})
        task.aggregation.assert_not_called()

    def test_aggregation_setup_failure_keeps_all_decisions_in_failure_artifacts(self):
        directory, _ = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        task.aggregation.side_effect = RuntimeError("native aggregation unavailable")
        output = self.root / "aggregation-failure"
        code, report = self.run_scoring_cli(directory, output, reference, task, scoring, response_class)
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report.get("results", [])), 23)
        self.assertEqual(report["scoring_errors"], 0)
        self.assertEqual(report["aggregation_errors"], 4)
        self.assertEqual(report["metrics_status"], "INCOMPLETE")
        self.assertEqual(report["first_mismatch"]["error"]["type"], "RuntimeError")
        reproduction = report["first_mismatch"]["input"]
        self.assertEqual(reproduction["scope"], "corpus")
        self.assertEqual([row["key"] for row in reproduction["records"]], self.keys)

    def test_cli_mismatch_saves_first_reproduction_input_and_all_decisions(self):
        directory, records = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        compute = task.metrics[0].compute_sample

        def wrong(doc, model_response):
            value = compute(doc, model_response)
            if doc.specific["ifeval_ko"]["key"] == 1203:
                value["inst_level_strict_acc"][1] = True
                value["prompt_level_strict_acc"] = True
                print("injected repeated-ID mismatch")
            return value

        task.metrics[0].compute_sample = wrong
        output = self.root / "decision-failure"
        code, report = self.run_scoring_cli(directory, output, reference, task, scoring, response_class)
        self.assertEqual(code, 1)
        self.assertEqual(len(report["results"]), 23)
        mismatch = report["first_mismatch"]
        self.assertEqual((mismatch["key"], mismatch["instruction_index"]), (1203, 1))
        self.assertEqual(mismatch["input"]["response"], next(row["response"] for row in records if row["key"] == 1203))
        self.assertIn("injected repeated-ID mismatch", (output / "run.log").read_text())

    def test_unexpected_execution_failure_preserves_progress_and_active_input(self):
        directory, records = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        calls = 0

        def interrupt():
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("unexpected execution failure")
            return {"python": 0, "langdetect": 0, "python_state_sha256": "test"}

        output = self.root / "interrupted-failure"
        with patch.object(self.verifier, "reset_seeds", side_effect=interrupt, create=True):
            code, report = self.run_scoring_cli(directory, output, reference, task, scoring, response_class)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["first_mismatch"]["input"]["response"], records[1]["response"])
        self.assertEqual(report["execution_error_input"]["key"], records[1]["key"])
        self.assertEqual(report["metrics"], {})
        self.assertEqual(report["metrics_status"], "INCOMPLETE")

    def test_interruption_preserves_earlier_mismatch_and_separate_execution_input(self):
        directory, records = self.make_run()
        reference, task, scoring, response_class, _ = self.engines()
        direct = scoring.score_response

        def different(prompt, ids, kwargs, text):
            value = direct(prompt, ids, kwargs, text)
            if prompt == records[0]["prompt"]:
                value["inst_level_strict_acc"][0] = False
                value["prompt_level_strict_acc"] = all(value["inst_level_strict_acc"])
            return value

        scoring.score_response = different
        calls = 0

        def interrupt():
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("interrupted after an earlier mismatch")
            return {"python": 0, "langdetect": 0, "python_state_sha256": "test"}

        output = self.root / "mismatch-then-interruption"
        with patch.object(self.verifier, "reset_seeds", side_effect=interrupt):
            code, report = self.run_scoring_cli(directory, output, reference, task, scoring, response_class)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "ERROR")
        first = report["first_mismatch"]
        self.assertEqual(first["key"], records[0]["key"])
        self.assertEqual(first["source"], "direct_ported")
        self.assertEqual(first["instruction_index"], 0)
        self.assertEqual(first["input"], {**self.verifier.raw_input(records[0]), "response": records[0]["response"]})
        self.assertEqual({key: value for key, value in first.items() if key != "input"}, report["mismatches"][0])
        self.assertEqual(
            report["execution_error_input"],
            {**self.verifier.raw_input(records[1]), "response": records[1]["response"]},
        )
        self.assertEqual(report["error"]["message"], "interrupted after an earlier mismatch")
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["metrics_status"], "INCOMPLETE")
        self.assertEqual(report["metrics"], {})

    def test_invalid_input_writes_failure_artifacts_without_loading_scorers(self):
        directory, _ = self.make_run()
        (directory / "responses.jsonl").write_text("tampered\n")
        output = self.root / "failure-report"
        with (
            patch.object(self.verifier, "load_reference") as loader,
            patch.dict(os.environ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = self.verifier.main(self.cli_args(directory, output))
        self.assertEqual(code, 1)
        loader.assert_not_called()
        report = json.loads((output / "model-response-parity.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["stage"], "input")
        for line in (output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            self.assertEqual(self.verifier.sha(output / name), digest)

    def test_existing_and_nested_output_directories_are_refused(self):
        directory, _ = self.make_run()
        for output in (directory, directory / "new-output"):
            with self.subTest(output=output), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    self.verifier.main(self.cli_args(directory, output))
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse((output / "model-response-parity.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

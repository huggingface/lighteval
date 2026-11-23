"""Generator compatibility and failure accounting tests without a model or MPS."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/ifeval_ko/generate_responses.py"


class Tokens:
    shape = (1, 1)

    def __getitem__(self, index):
        return self

    def tolist(self):
        return [2]


class Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=Tokens())

    def to(self, device):
        return self


class Terminal(io.StringIO):
    def __init__(self, attempts, *, success_at=None, progress_at=None, fail_summary=False, failure="write", total=23):
        super().__init__()
        self.attempts = attempts
        self.success_at = success_at
        self.progress_at = progress_at
        self.fail_summary = fail_summary
        self.failure = failure
        self.total = total

    def write(self, message):
        success_error = self.failure == "write" and self.attempts() == self.success_at and message.startswith("  ")
        progress_error = self.progress_at is not None and message.startswith(f"[{self.progress_at}/{self.total}]")
        if success_error or progress_error or (self.fail_summary and message.startswith("{")):
            raise BrokenPipeError("terminal write failed")
        return super().write(message)

    def flush(self):
        if self.failure == "flush" and self.attempts() == self.success_at:
            raise BrokenPipeError("terminal flush failed")
        return super().flush()


class GenerationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("_test_ifeval_ko_generator", SCRIPT)
        self.generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.generator)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.selection = self.root / "selection.json"
        self.selection.write_text(json.dumps({"keys": self.generator.KEYS}), encoding="utf-8")
        self.run_index = 0
        self.rows = []
        instruction_index = 0
        for index, key in enumerate(self.generator.KEYS + list(range(100000, 100319))):
            count = 2 if index < 17 or 23 <= index < 145 else 1
            ids = [f"type-{i % 21}" for i in range(instruction_index, instruction_index + count)]
            self.rows.append(
                {"key": key, "prompt": f"Prompt {key}", "instruction_id_list": ids, "kwargs": [{}] * count}
            )
            instruction_index += count

    def run_generation(
        self,
        *,
        fail_generation_at=None,
        fail_log_at=None,
        fail_progress_at=None,
        fail_summary=False,
        terminal_failure=None,
        legacy_loader=False,
        scope="selected",
    ):
        self.run_index += 1
        output = self.root / f"run-{self.run_index}"
        attempts = 0

        def generate(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == fail_generation_at:
                raise RuntimeError("generation failed")
            return types.SimpleNamespace(sequences=Tokens())

        model = types.SimpleNamespace(
            generation_config=types.SimpleNamespace(eos_token_id=2, to_dict=lambda: {}),
            config=types.SimpleNamespace(max_position_embeddings=32768),
            parameters=lambda: iter([types.SimpleNamespace(dtype="torch.float16")]),
            generate=generate,
        )
        model.to = lambda device: model
        model.eval = lambda: model

        # The older loader accepts torch_dtype but has no dtype keyword.
        def load_legacy(model_id, *, revision, torch_dtype, trust_remote_code):
            self.assertIs(torch_dtype, sentinel.float16)
            return model

        load_model = Mock(side_effect=load_legacy) if legacy_loader else Mock(return_value=model)
        tokenizer = types.SimpleNamespace(
            chat_template="test template",
            eos_token_id=2,
            pad_token_id=0,
            apply_chat_template=lambda *args, **kwargs: "test input",
            decode=lambda *args, **kwargs: "test response",
        )
        tokenize = Mock(return_value=Inputs())
        tokenize.configure_mock(**vars(tokenizer))
        modules = {
            "torch": types.SimpleNamespace(
                backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
                float16=sentinel.float16,
                mps=types.SimpleNamespace(synchronize=lambda: None),
                inference_mode=contextlib.nullcontext,
            ),
            "datasets": types.SimpleNamespace(load_dataset=lambda *args, **kwargs: self.rows),
            "transformers": types.SimpleNamespace(
                AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_model),
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenize),
                set_seed=lambda seed: None,
            ),
        }

        terminal = Terminal(
            lambda: attempts,
            success_at=fail_log_at,
            progress_at=fail_progress_at,
            fail_summary=fail_summary,
            failure=terminal_failure or "write",
            total=self.generator.SCOPES[scope][0],
        )
        argv = [str(SCRIPT), "--output-dir", str(output)]
        if scope == "selected":
            argv += ["--selection", str(self.selection)]
        else:
            argv += ["--scope", scope]
        previous_directory = Path.cwd()
        try:
            os.chdir(SCRIPT.parents[2])
            with (
                patch.dict(sys.modules, modules),
                patch.object(sys, "argv", argv),
                patch.object(self.generator, "command", return_value="0"),
                patch.object(self.generator, "version", return_value="test"),
                contextlib.redirect_stdout(terminal),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = self.generator.main()
        finally:
            os.chdir(previous_directory)
        manifest = json.loads((output / "manifest.json").read_text())
        for line in (output / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            self.assertNotEqual(name, "SHA256SUMS")
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
        for name, expected in manifest["artifact_sha256"].items():
            self.assertNotIn(name, ("manifest.json", "SHA256SUMS"))
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
        path = output / "responses.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return code, manifest, records

    def assert_accounting(self, manifest, records, expected, total=23):
        counts = tuple(manifest[name] for name in ("generated_count", "generation_error_count", "unattempted_count"))
        self.assertEqual(counts, expected)
        self.assertEqual(sum(counts), total)
        self.assertTrue(all(count >= 0 for count in counts))
        self.assertEqual(manifest["generated_count"], sum(record["status"] == "generated" for record in records))
        self.assertEqual(manifest["generation_error_count"], sum(record["status"] == "error" for record in records))
        self.assertEqual(manifest["unattempted_count"], total - len(records))

    def test_full_split_keeps_dataset_order_without_selection_file(self):
        self.selection.unlink()
        code, manifest, records = self.run_generation(scope="full", legacy_loader=True)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["evaluation_scope"], "full")
        self.assertEqual(manifest["expected_count"], 342)
        self.assertEqual(manifest["instruction_occurrences"], 481)
        self.assertIsNone(manifest["selection_source"])
        for row, record in zip(self.rows, records):
            self.assertEqual({field: record[field] for field in row}, row)
        self.assertEqual([row["key"] for row in records], [row["key"] for row in self.rows])
        self.assert_accounting(manifest, records, (342, 0, 0), total=342)

    def test_full_split_failure_counts_do_not_use_selected_denominator(self):
        code, manifest, records = self.run_generation(scope="full", fail_generation_at=200)
        self.assertEqual(code, 1)
        self.assert_accounting(manifest, records, (199, 1, 142), total=342)
        code, manifest, records = self.run_generation(scope="full", fail_log_at=342)
        self.assertEqual(code, 1)
        self.assertEqual(manifest["error_stage"], "logging")
        self.assert_accounting(manifest, records, (342, 0, 0), total=342)

    def test_full_split_rejects_selection_and_misaligned_unselected_row(self):
        with self.assertRaisesRegex(ValueError, "not a selection file"):
            self.generator.select_rows(self.rows, self.generator.KEYS, scope="full")
        self.rows[-1]["kwargs"] = []
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.generator.select_rows(self.rows, scope="full")

    def test_full_scope_cli_rejects_selection_before_creating_output(self):
        output = self.root / "rejected"
        argv = [str(SCRIPT), "--scope", "full", "--selection", str(self.selection), "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.generator.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(output.exists())

    def test_success_with_legacy_loader_keyword(self):
        code, manifest, records = self.run_generation(legacy_loader=True)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "responses_generated")
        self.assertEqual(manifest["logging_errors"], [])
        self.assert_accounting(manifest, records, (23, 0, 0))

    def test_success_log_failure_preserves_saved_response(self):
        for index in (1, 23):
            with self.subTest(index=index):
                code, manifest, records = self.run_generation(fail_log_at=index)
                self.assertEqual(code, 1)
                self.assertEqual(manifest["error"]["type"], "BrokenPipeError")
                self.assertEqual(manifest["error_stage"], "logging")
                self.assertEqual(manifest["logging_errors"][0]["type"], "BrokenPipeError")
                self.assertEqual(records[-1]["status"], "generated")
                self.assertEqual(records[-1]["response"], "test response")
                self.assert_accounting(manifest, records, (index, 0, 23 - index))

    def test_progress_log_failure_does_not_create_generation_error(self):
        for index in (1, 7):
            with self.subTest(index=index):
                code, manifest, records = self.run_generation(fail_progress_at=index)
                self.assertEqual(code, 1)
                self.assertEqual(manifest["error_stage"], "logging")
                self.assertEqual(len(manifest["logging_errors"]), 1)
                self.assert_accounting(manifest, records, (index - 1, 0, 24 - index))

    def test_real_tee_terminal_failure_preserves_all_saved_responses(self):
        for failure in ("write", "flush"):
            with self.subTest(failure=failure):
                code, manifest, records = self.run_generation(fail_log_at=23, terminal_failure=failure)
                self.assertEqual(code, 1)
                self.assertEqual(manifest["error_stage"], "logging")
                self.assertEqual(manifest["logging_errors"][0]["type"], "BrokenPipeError")
                self.assertTrue(all(record["status"] == "generated" for record in records))
                self.assert_accounting(manifest, records, (23, 0, 0))

    def test_summary_log_failure_is_recorded_after_generation(self):
        code, manifest, records = self.run_generation(fail_summary=True)
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "error")
        self.assertEqual(manifest["error_stage"], "logging")
        self.assertEqual(len(manifest["logging_errors"]), 1)
        self.assert_accounting(manifest, records, (23, 0, 0))

    def test_generation_failure_preserves_completed_and_unattempted_counts(self):
        for index in (1, 7):
            with self.subTest(index=index):
                code, manifest, records = self.run_generation(fail_generation_at=index)
                self.assertEqual(code, 1)
                self.assertEqual(manifest["error"]["type"], "RuntimeError")
                self.assertEqual(manifest["logging_errors"], [])
                self.assert_accounting(manifest, records, (index - 1, 1, 23 - index))


if __name__ == "__main__":
    unittest.main(verbosity=2)

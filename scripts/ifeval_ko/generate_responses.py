"""Generate selected or full-split IFEval-Ko responses without running a scorer."""

import argparse
import contextlib
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATA_REVISION = "54199e3801116897697babf341865741dcd06fc8"
KEYS = [
    3743,
    136,
    3518,
    1233,
    3445,
    3442,
    1262,
    1082,
    1127,
    3710,
    122,
    3563,
    1107,
    3073,
    3311,
    3750,
    1203,
    2142,
    1268,
    3538,
    1928,
    2035,
    1800,
]
STOPS = ["<end_of_turn>", "<|im_end|>"]
SCOPES = {"selected": (23, 40), "full": (342, 481)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def log_message(manifest, message):
    try:
        print(message, flush=True)
    except Exception as error:
        manifest.setdefault("logging_errors", []).append({"type": type(error).__name__, "message": str(error)})
        raise


def write_artifacts(output_dir, manifest):
    artifacts = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    manifest["artifact_sha256"] = {path.name: digest(path) for path in artifacts}
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    sums = "".join(f"{digest(path)}  {path.name}\n" for path in sorted([*artifacts, manifest_path]))
    (output_dir / "SHA256SUMS").write_text(sums, encoding="utf-8")


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def select_rows(rows, keys=None, scope="selected"):
    by_key = {row["key"]: row for row in rows}
    if len(rows) != 342 or len(by_key) != 342:
        raise ValueError("Expected 342 source rows with unique keys.")
    if sum(len(row["instruction_id_list"]) for row in rows) != 481:
        raise ValueError("Expected 481 source instruction occurrences.")
    if scope == "full":
        if keys is not None:
            raise ValueError("Full-split generation uses the pinned dataset order, not a selection file.")
        selected = rows
    elif scope == "selected":
        if keys != KEYS or len(set(keys)) != 23:
            raise ValueError("Selection must match the agreed 23 keys in their recorded order.")
        selected = [by_key[key] for key in keys]
    else:
        raise ValueError(f"Unknown evaluation scope: {scope}")
    expected_count, expected_instructions = SCOPES[scope]
    if (
        len(selected) != expected_count
        or sum(len(row["instruction_id_list"]) for row in selected) != expected_instructions
    ):
        raise ValueError(f"Expected {expected_count} prompts and {expected_instructions} instructions.")
    if len({iid for row in selected for iid in row["instruction_id_list"]}) != 21:
        raise ValueError("Expected all 21 instruction types in the selection.")
    if any(len(row["instruction_id_list"]) != len(row["kwargs"]) for row in selected):
        raise ValueError("Instruction/kwargs alignment differs.")
    return selected


def extract_response(tokenizer, ids, config):
    raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    response = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for stop in STOPS:
        response = response.split(stop, 1)[0]
    eos_ids = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    matches = [stop for stop in STOPS if stop in raw]
    if ids and ids[-1] in eos_ids:
        reason = "eos"
    elif matches:
        reason = "stop_string"
    elif len(ids) >= config.max_new_tokens:
        reason = "length"
    else:
        reason = "unknown"
    return {
        "raw_generated_text": raw,
        "response": response,
        "scoring_response": response,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "output_token_ids": ids,
        "generated_token_count": len(ids),
        "finish_reason": reason,
        "matched_stop_strings": matches,
    }


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.logfile.write(text)
        self.logfile.flush()
        return self.terminal.write(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def generate(args, manifest):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; this run will not silently switch to CPU.")
    dataset = load_dataset("allganize/IFEval-Ko", "default", split="train", revision=DATA_REVISION)
    if args.scope == "full":
        rows = select_rows(list(dataset), scope="full")
        selection = {"scope": "full", "order": "pinned train split", "keys": [row["key"] for row in rows]}
        write_json(args.output_dir / "selection.json", selection)
    else:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        rows = select_rows(list(dataset), selection["keys"])
        shutil.copyfile(args.selection, args.output_dir / "selection.json")
    write_json(args.output_dir / "selected-samples.json", rows)
    manifest.update(
        selected_keys=selection["keys"],
        instruction_occurrences=sum(len(row["instruction_id_list"]) for row in rows),
        code_revision=command("git", "rev-parse", "HEAD"),
        worktree_status=command("git", "status", "--short"),
        chip=command("sysctl", "-n", "machdep.cpu.brand_string"),
        memory_bytes=int(command("sysctl", "-n", "hw.memsize")),
        python=platform.python_version(),
        macos=platform.mac_ver()[0],
        architecture=platform.machine(),
        package_versions={
            name: version(name)
            for name in ("lighteval", "torch", "transformers", "accelerate", "datasets", "huggingface_hub", "numpy")
        },
        selection_source=str(args.selection.resolve()) if args.selection is not None else None,
    )
    frozen = command(sys.executable, "-m", "pip", "freeze", "--exclude-editable")
    (args.output_dir / "requirements.lock.txt").write_text(frozen + "\n", encoding="utf-8")
    write_json(args.output_dir / "manifest.json", manifest)
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    log_message(manifest, "Loading the pinned Qwen model on MPS...")
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, trust_remote_code=False
        )
        .to("mps")
        .eval()
    )
    config = copy.deepcopy(model.generation_config)
    config.max_new_tokens = 1280
    config.do_sample = False
    config.temperature = 0.0
    config.top_p = 1.0
    config.top_k = 0
    config.num_beams = 1
    config.num_return_sequences = 1
    config.stop_strings = STOPS
    config.return_dict_in_generate = True
    manifest.update(
        generation_config=config.to_dict(),
        actual_dtype=str(next(model.parameters()).dtype),
        chat_template=tokenizer.chat_template,
        tokenizer_eos_token_id=tokenizer.eos_token_id,
        tokenizer_pad_token_id=tokenizer.pad_token_id,
        model_max_position_embeddings=model.config.max_position_embeddings,
    )
    write_json(args.output_dir / "manifest.json", manifest)
    with (args.output_dir / "responses.jsonl").open("x", encoding="utf-8") as responses:
        for index, row in enumerate(rows, 1):
            log_message(manifest, f"[{index}/{manifest['expected_count']}] Generating key {row['key']}...")
            record = copy.deepcopy(row)
            record["status"] = "started"
            try:
                messages = [{"role": "user", "content": row["prompt"]}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("mps")
                record.update(rendered_input=rendered, input_token_ids=inputs["input_ids"][0].tolist())
                if len(record["input_token_ids"]) + 1280 > model.config.max_position_embeddings:
                    raise ValueError("Prompt plus output budget exceeds context length; no input truncation applied.")
                torch.mps.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=config, tokenizer=tokenizer)
                torch.mps.synchronize()
                ids = generated.sequences[0, inputs["input_ids"].shape[1] :].tolist()
                record.update(extract_response(tokenizer, ids, config))
                record.update(status="generated", generation_seconds=time.perf_counter() - started)
            except (Exception, KeyboardInterrupt) as error:
                record.update(status="error", error={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                responses.write(json.dumps(record, ensure_ascii=False) + "\n")
                responses.flush()
                os.fsync(responses.fileno())
                if record["status"] == "generated":
                    manifest["generated_count"] += 1
                    manifest["length_limited_count"] += int(record["finish_reason"] == "length")
                    manifest["empty_response_count"] += int(not record["response"].strip())
                elif record["status"] == "error":
                    manifest["generation_error_count"] += 1
                write_json(args.output_dir / "manifest.json", manifest)
            log_message(
                manifest, f"  {len(ids)} tokens; finish={record['finish_reason']}; {record['generation_seconds']:.1f}s"
            )
    manifest["status"] = "responses_generated"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="selected", help="Selected 23 prompts or all 342 prompts")
    parser.add_argument("--selection", type=Path, help="Selection JSON for --scope selected only")
    args = parser.parse_args()
    if args.scope == "full" and args.selection is not None:
        parser.error("--selection cannot be used with --scope full; the pinned dataset order is required.")
    if args.scope == "selected" and args.selection is None:
        args.selection = Path("../k-llm-eval-sprint/teams/team-c-ifeval-ko/evidence/sample-coverage/selection.json")
    if not Path("pyproject.toml").is_file():
        parser.error("Run from the Lighteval repository root.")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose another name to preserve earlier runs.")
    return args


def main():
    args = parse_args()
    expected_count, _ = SCOPES[args.scope]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output_dir / "generate_responses.py")
    manifest = {
        "status": "running",
        "scope": f"{args.scope} {expected_count} prompts; response generation only",
        "evaluation_scope": args.scope,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_id": "allganize/IFEval-Ko",
        "dataset_revision": DATA_REVISION,
        "dataset_subset": "default",
        "dataset_split": "train",
        "num_fewshots": 0,
        "backend": "transformers.AutoModelForCausalLM.generate",
        "device": "mps",
        "requested_dtype": "float16",
        "batch_size": 1,
        "seed": 0,
        "expected_count": expected_count,
        "generated_count": 0,
        "generation_error_count": 0,
        "logging_errors": [],
        "scored_count": 0,
        "scoring_status": "not_run",
        "native_lighteval_inference_run": False,
        "length_limited_count": 0,
        "empty_response_count": 0,
        "decode_policy": "skip special tokens and remove stop strings; no strip, NFC, or reasoning cleanup",
        "command": [sys.executable, *sys.argv],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    code = 0
    started = time.perf_counter()
    with (args.output_dir / "run.log").open("x", encoding="utf-8") as logfile:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, logfile)),
            contextlib.redirect_stderr(Tee(sys.stderr, logfile)),
        ):
            try:
                generate(args, manifest)
            except (Exception, KeyboardInterrupt) as error:
                manifest.update(status="error", error={"type": type(error).__name__, "message": str(error)})
                if manifest["logging_errors"]:
                    manifest["error_stage"] = "logging"
                code = 1
                try:
                    traceback.print_exc()
                except Exception as logging_error:
                    manifest["logging_errors"].append(
                        {"type": type(logging_error).__name__, "message": str(logging_error)}
                    )
            manifest.update(
                ended_at_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter() - started,
                unattempted_count=expected_count - manifest["generated_count"] - manifest["generation_error_count"],
            )
    write_artifacts(args.output_dir, manifest)
    if not manifest["logging_errors"]:
        summary = json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "expected_count",
                    "generated_count",
                    "generation_error_count",
                    "unattempted_count",
                    "scored_count",
                    "length_limited_count",
                    "empty_response_count",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
        try:
            log_message(manifest, summary)
            log_message(manifest, f"Saved: {args.output_dir.resolve()}")
        except Exception as error:
            if "error" not in manifest:
                manifest.update(error={"type": type(error).__name__, "message": str(error)}, error_stage="logging")
            manifest["status"] = "error"
            code = 1
            write_artifacts(args.output_dir, manifest)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

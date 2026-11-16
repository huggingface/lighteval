# IFEval-Ko: selected and full-split response generation

Generate and save one Qwen response per selected prompt so the team can pass
the same saved responses to the original Ko evaluator and registered Lighteval
task. This stage runs Transformers generation only. It does not invoke either
scorer or establish model-response scoring parity.

The default remains the agreed 23-prompt selection. `--scope full` generates all
342 prompts and 481 instruction occurrences at the same pinned dataset revision,
using the same model, chat template, decoding settings and MPS backend below.

## Recorded run

| Item | Value |
| --- | --- |
| Model and tokenizer | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model/tokenizer revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Dataset | `allganize/IFEval-Ko`, `default` / `train` |
| Dataset revision | `54199e3801116897697babf341865741dcd06fc8` |
| Selection | 23 prompts, 40 instruction occurrences |
| Backend | Transformers `AutoModelForCausalLM.generate` |
| Device and requested dtype | Apple MPS, float16 |
| Batch size / responses per prompt | 1 / 1 |
| Few-shot examples | 0 |
| Sampling / temperature | `do_sample=False` / `temperature=0.0` |
| Output limit | 1,280 new tokens per prompt |
| Declared stop strings | `<end_of_turn>`, `<\|im_end\|>` |
| Seed | 0 |
| Generated / generation errors / unattempted | 23 / 0 / 0 |
| Empty responses | 0 |
| Token-limit termination | key `1127` |
| Total script runtime | approximately 185.6 seconds |
| Responses scored | 0 |

The complete generation configuration, including inherited model settings and
EOS/pad IDs, actual dtype, environment versions, execution timestamps and Git
revision are recorded in `manifest.json`. The official tokenizer chat template
is applied once; its rendered input and input token IDs are saved for each
prompt. No custom system message is supplied; any default system text comes
from the pinned tokenizer template.

The model's sampling defaults are explicitly overridden. Transformers may warn
that `temperature` and `top_k` are ignored because sampling is disabled.

Selected keys, in generation order:

```json
[3743, 136, 3518, 1233, 3445, 3442, 1262, 1082, 1127, 3710, 122,
 3563, 1107, 3073, 3311, 3750, 1203, 2142, 1268, 3538, 1928, 2035, 1800]
```

## Run from the repository root

Use the team's merged `team-c/ifeval-ko` code and an activated Python 3.11
environment with the required dependencies installed. The task merge commit is
`bc7bc0beaee028e0e2f16a734c36048c121609c6`; the actual run's code revision is in
the manifest, and an exact generator snapshot is included in the run artifacts.

The default selection path expects the operations repository next to the code
repository:

```text
Projects/
  k-llm-eval-sprint/
  lighteval-ifeval-ko/
```

```bash
python scripts/ifeval_ko/generate_responses.py \
  --output-dir outputs/ifeval_ko/qwen-23-001
```

On macOS, prefix the command with `caffeinate -i` to prevent idle sleep during
generation. An existing output directory is rejected; use a new name for another
run. The script checks the selection against the recorded 23 keys, 40
instruction occurrences, and 21 instruction types before loading the model.

## Generate the full 342-prompt split

Run from the repository root in the same Python environment:

```bash
python scripts/ifeval_ko/generate_responses.py \
  --scope full --output-dir outputs/ifeval_ko/qwen-342-001
```

On macOS, prefix the command with `caffeinate -i` to keep the machine awake.
The full scope does not need the operations repository or a selection file.
`--selection` is rejected with `--scope full` so a partial selection cannot
accidentally be labelled as full-split evaluation.

All 342 original rows are processed in the pinned dataset's existing `train`
order, without sorting keys, modifying prompts/kwargs, reusing the old 23
responses, or dropping token-limited or empty generated responses. The script
checks the source size, 481 instruction occurrences, all 21 instruction types,
unique keys and instruction/kwargs alignment before loading the model.

The seven-file artifact contract below is unchanged. For full runs,
`selection.json` is generated from the source order with `scope: full`, and
`selected-samples.json` contains all 342 original rows. The manifest records
`evaluation_scope: full`, `expected_count: 342` and
`instruction_occurrences: 481`; `selection_source` is null. Generation and log
failures retain the same accounting policy with a denominator of 342.

Full-split scoring uses the same saved responses in all three evaluator paths:

```bash
PYTHONDONTWRITEBYTECODE=1 python tests/ifeval_ko/verify_model_responses.py \
  --scope full --run-dir outputs/ifeval_ko/qwen-342-001 \
  --audit-dir "$KO_AUDIT" --source-root "$KO_SOURCES" \
  --output-dir outputs/ifeval_ko/qwen-342-parity-001
```

Prepare the pinned audit sources and NLTK data as described in
[`tests/ifeval_ko/README.md`](../../tests/ifeval_ko/README.md#reproduction).
The registered task registry imports other built-in tasks; its first load can
also download the existing TinyBenchmarks cache into the checkout. Prepare this
cache during setup when validation must run offline. Cache/output files are not
part of the source changes.

For archive verification, also pass `--archive` and the independently recorded
`--archive-sha256` digest, as for selected runs. Archive checking is optional and
reported separately; source inputs and all run-file hashes are always checked.

### Recorded full-split run

Results for `qwen-342-001`, including generation accounting, token-limited keys,
tokenizer replay and scoring comparisons, are maintained in the
[full-split validation record](../../tests/ifeval_ko/README.md#validation-with-full-split-saved-responses-342-prompts).

Download the response and parity ZIPs from the
[public attachments and checksums](../../tests/ifeval_ko/README.md#public-attachments-and-checksums).
These files are attached to PR #7 rather than checked into the repository.

## Saved artifacts and integrity

The run directory contains:

| File | Contents |
| --- | --- |
| `selection.json` | Copy of the selection file used by this run |
| `selected-samples.json` | Original selected rows, including ordered IDs and unmodified kwargs |
| `responses.jsonl` | One record per attempted prompt, with source fields, rendered input, input/output token IDs, raw decoded text, scoring response, termination reason and generation time or error |
| `manifest.json` | Revisions, actual generation settings, environment, stage counts and artifact hashes |
| `requirements.lock.txt` | Installed dependency versions, excluding editable packages |
| `generate_responses.py` | Exact script snapshot used by the run |
| `run.log` | Execution log |
| `SHA256SUMS` | SHA-256 checksums for the seven files above |

The saved run passed checksum verification for all seven files. Additional
checks confirmed 23 unique keys in the selected order, 23 generated responses,
40 instructions, unchanged original input fields, matching response hashes,
and identical `response` / `scoring_response` fields. There were no generation
errors, unattempted prompts or empty responses.

Run checksum verification from inside the extracted run directory:

```bash
shasum -a 256 -c SHA256SUMS
```

Preserve the original run directory. Package it as a PR attachment; the
repository ignores `outputs/` and JSON runtime artifacts. The generator and
this README are source changes that can be committed separately.

To reproduce generation in a fresh checkout and Python 3.11 environment, point
`KO_RUN_DIR` at the extracted original run directory:

```bash
python -m pip install -e . -r "$KO_RUN_DIR/requirements.lock.txt"
python "$KO_RUN_DIR/generate_responses.py" \
  --selection "$KO_RUN_DIR/selection.json" \
  --output-dir outputs/ifeval_ko/qwen-23-reproduction-001
```

Run these commands from the Lighteval repository root. Reproduction requires an
MPS-capable Mac. File checksums verify saved artifacts; new runs have different
timestamps and logs, and cross-environment generation is not guaranteed to be
byte-identical.

## Response contract and remaining work

For both evaluators, use each generated record's `response` or equivalent
`scoring_response` verbatim, together with its original `key`, `prompt`,
`instruction_id_list` and `kwargs`. Special tokens and declared stop strings
are removed during response extraction. No whitespace stripping, NFC
normalization, Markdown cleanup, or reasoning removal is applied here. The
original generated token IDs and decoded text including special tokens are
preserved separately.

Token-limit termination is retained as a model response, not converted into a
generation error or removed from the sample. Key `1127` must remain in the
cross-evaluation input. A blank response, if generated in another run, is also
retained. Runtime exceptions are saved as errors and stop the run; they are not
replaced with blank successful responses.

The one-item hardware smoke test is separate from the recorded 23-prompt run.
The generation stage reports no scoring result. Compare identical saved text
through original Ko, direct ported scoring and the registered task with the
matching verifier scope, and investigate every mismatch. Saved-response scoring
and native Lighteval model inference are separate validation stages.

# IFEval-Ko scoring

`src/lighteval/tasks/tasks/ifeval_ko/scoring.py` provides the independent Ko
strict/loose scorer. **Both direct scorer and registered-task regression pass
all 156 fixed-response cases**, matching the predeclared expectations, stored
original Ko results and a live run of the pinned original evaluator.

## Current validation status

- Direct scorer with the team's ported checkers: **156/156 cases**, strict **247/247 instructions + 156/156 prompts**,
  loose **247/247 instructions + 156/156 prompts**. Missing cases, unexpected
  scoring errors, decision differences and call-trace differences: **0**.
- Registered `ifeval_ko|0` task: **156/156 cases**, the same instruction/prompt
  decisions and four metrics through Registry, Doc conversion, `compute_sample`
  and native corpus aggregation. The scorer is also called directly in this run.
- Auxiliary inputs: **18/18** for both paths, each mode **10 normal returns + 8 expected exceptions**.
  These are excluded from the main metric denominators. Expected exceptions are
  contract checks, not successful normal scores.
- Policy/verifier tests: **11 passed**. Reference audit tests: **25 passed**.
- Ruff check and format check pass with Ruff `0.11.10`.
- Full pinned dataset formatting: **342/342 Docs**, preserving the original
  keys, prompts, kwargs and **481 ordered instruction occurrences**. This is
  dataset formatting validation, not model response scoring.
- Saved Qwen responses: the original Ko evaluator, ported scorer and registered
  task agree on strict and loose judgments for **23/23 prompts** and
  **40/40 instruction occurrences**. All four native corpus metrics match the
  original Ko results exactly. The saved response verifier's unit tests pass,
  including full-split controls; the generator tests also pass.
  See [validation with saved model responses](#validation-with-saved-model-responses-23-selected-prompts).
- Full saved Qwen split: generation results, strict/loose agreement and all four
  metrics are recorded in [full-split validation](#validation-with-full-split-saved-responses-342-prompts).

The following table shows the **156 fixed response regression results**.

| Metric | Original Ko | Ported scorer | Registered task | Absolute error |
| --- | --- | --- | --- | --- |
| `prompt_level_strict_acc` | 71/156 | 71/156 | 71/156 | 0 |
| `inst_level_strict_acc` | 133/247 | 133/247 | 133/247 | 0 |
| `prompt_level_loose_acc` | 79/156 | 79/156 | 79/156 | 0 |
| `inst_level_loose_acc` | 142/247 | 142/247 | 142/247 | 0 |

The 156-case regression used the integrated task implementation at
`21faccc0f27d35da37dd849894f37b0337b43edb`. The reference uses the audit's explicit
`isolated-source` import mode, retaining its original imports and source bytes;
the archived reference run used `full-package`. The direct scorer and registered
task use the team's ported checkers, without a reference fallback.

The registered-task results attached to
[PR #2](https://github.com/Hugging-Face-KREW/lighteval/pull/2#issuecomment-6718587652)
also matched by `case_id`, instruction index, and input/response hashes.
This optional attachment comparison is separate from locally executing the task.

## Remaining validation

English and French runtime regression and native Lighteval model inference remain
pending. The verifier supports both the default 23-prompt selection and explicit
`--scope full` for all 342 prompts and 481 instruction occurrences. Use the same
saved assistant responses for the original Ko evaluator and both Lighteval paths,
and record the generation settings and any errors. Full-split execution results
are recorded separately from the historical 23-prompt results below.

`verify_registered_task.py` checks dataset formatting through the task API.
Output from `lighteval tasks inspect` is tracked separately as PR evidence.

## Validation with saved model responses (23 selected prompts)

[`verify_model_responses.py`](verify_model_responses.py) evaluates the 23 saved
`qwen-23-001` responses attached to [PR #5](https://github.com/Hugging-Face-KREW/lighteval/pull/5).
It runs in a CPU Lighteval environment without loading model weights or
generating new responses. Each saved response is evaluated through three paths:

1. The original Ko evaluator at revision
   `54199e3801116897697babf341865741dcd06fc8`.
2. A direct call to the ported `scoring.score_response` function.
3. The registered `ifeval_ko|0` task, using `Registry`, Doc formatting,
   `ModelResponse`, `compute_sample` and native corpus aggregation.

The original Ko evaluator runs on the saved responses, and its actual outputs
serve as the reference. This comparison is separate from the 156 fixed response
regression cases and their predefined expectations.

All three paths agreed on every strict and loose judgment: **23/23 prompts**
and **40/40 instruction occurrences** for each mode. These agreement counts
measure consistency between the evaluation paths. The table below shows the
model's scores. Numerators and denominators match across all three paths, and
each native corpus metric has an absolute error of **0** relative to the original
Ko result, within the required tolerance of `1e-12`.

| Metric | Value in all three paths | Native absolute error |
| --- | --- | --- |
| `prompt_level_strict_acc` | 6/23 | 0 |
| `inst_level_strict_acc` | 19/40 | 0 |
| `prompt_level_loose_acc` | 9/23 | 0 |
| `inst_level_loose_acc` | 23/40 | 0 |

There were no missing records, scoring errors, aggregation errors or unresolved
differences. The response for key `1127`, which reached the token limit, remains
included in the comparison and metric denominators.

The verifier passes each saved response verbatim, without additional whitespace
removal, normalization or Markdown cleanup. The original Ko scoring rules,
including NFC normalization and loose matching, still run inside each evaluator.
Unicode line separators inside response strings are preserved when reading JSONL.
Repeated instruction IDs are preserved by comparing
`(key, instruction_index, instruction_id)`, where `instruction_index` starts at
zero. The JSON report includes strict and loose judgments from all three paths
for each of the 40 instruction occurrences.

Immediately before each of the 69 scoring calls, the verifier calls
`random.seed(0)` and sets `langdetect.DetectorFactory.seed` to `0`. The report
records the input and response hashes, seed values and Python RNG state hash
for every call. Before scoring, the verifier checks:

- Checksums for all seven saved run files and the artifact hashes in the manifest.
- The agreed 23 keys and their order, plus 40 instruction occurrences covering
  21 instruction types.
- Each response hash and the original prompt, key, ordered instruction IDs and
  kwargs against the Ko dataset at the pinned revision.
- The registered Doc's source fields, using hashes that preserve JSON types so
  that a change from `1` to `true`, for example, is detected.

When the original ZIP and its published digest are supplied, the verifier also
checks the archive hash and compares the extracted files with the archive
contents byte for byte. It records the generation revision and archived script
hash separately from the evaluation code. The archived generator may predate
the fixes in the current checkout.

Use the audit and source directories described in [Reproduction](#reproduction).
Download and extract the original attachment, then choose a new output directory.
Run the following command from the Lighteval repository root:

```bash
KO_RUN_DIR=/path/to/qwen-23-001
KO_ARCHIVE=/path/to/qwen-23-001.zip
KO_AUDIT=/path/to/pinned-checker-audit
KO_SOURCES=/path/to/restored-audit-sources
PYTHONDONTWRITEBYTECODE=1 python3 tests/ifeval_ko/verify_model_responses.py \
  --run-dir "$KO_RUN_DIR" \
  --archive "$KO_ARCHIVE" \
  --archive-sha256 a28e35bde3327a05efa9c255c228fa5ea2e711328e6e997c654d5f3b4f9e0e05 \
  --audit-dir "$KO_AUDIT" --source-root "$KO_SOURCES" \
  --output-dir outputs/ifeval_ko/qwen-23-parity-001
```

The published digest is available in the
[PR #5 ZIP checksum attachment](https://github.com/user-attachments/files/34706603/qwen-23-001.zip.sha256.txt).
The output directory contains `model-response-parity.json`,
`model-response-parity.md`, `run.log` and `SHA256SUMS`. The JSON report records
individual judgments, all four metrics with their numerators, denominators and
values, all mismatches and the first input needed to reproduce a failure. It also
records input and source hashes, code revisions, package versions and the
execution command. Original inputs are preserved, and an existing output
directory cannot be reused.

Missing, duplicate or reordered keys, altered inputs or hashes, generation
errors and scoring failures produce a nonzero exit status. This comparison
requires valid results for every input, so identical exceptions in all three
paths still count as failures. If a scoring call fails, the verifier keeps the
individual results but does not report metrics for a reduced set of prompts.
The expected denominators remain 23 prompts and 40 instruction occurrences.

If native aggregation fails during setup or computation, the report retains
the individual judgments and full numerators and denominators, with the error
recorded for each affected metric. A failure in corpus aggregation also includes
all 23 original inputs and responses for reproduction. An unexpected interruption
during scoring preserves completed records and the input being processed.
If a mismatch occurred earlier, its reproduction input remains in `first_mismatch`;
the interrupted input is saved separately as `execution_error_input`.
These failures produce diagnostic reports and a nonzero exit status.

A run with a logging failure can be evaluated if all 23 generated responses
were saved and pass the completeness checks. Successfully generated empty
responses are retained. Generation failures cannot be replaced with empty
responses to satisfy the expected record count.

The [verifier unit tests](../unit/tasks/test_ifeval_ko_model_responses.py) cover
input and archive integrity, repeated instruction IDs, incorrect denominators
and corpus aggregation, seed resets before each call, response and Doc changes,
scoring exceptions, interrupted execution and diagnostic reports. Run them with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/unit/tasks/test_ifeval_ko_model_responses.py
```

## Validation with full-split saved responses (342 prompts)

The recorded `qwen-342-001` run generated **342/342 responses** with **0 generation
errors, 0 unattempted prompts and 0 empty responses** in approximately **3,146.7
seconds (52.4 minutes)**. Its actual model/tokenizer revision, complete generation
configuration, float16 dtype, chat template, device, seed and batch size matched
the historical PR #5 run. All saved source rows matched the pinned Parquet.
The prompts shared with PR #5 were generated again; all 23 response hashes
matched the original saved responses.

The full run retained 12 token-limited responses, in source order:
`1127, 19, 2015, 2464, 2485, 2515, 2749, 2844, 3223, 337, 3439, 3606`.

Three-path replay passed: strict and loose each matched **342/342 prompt
judgments** and **481/481 instruction judgments**. The model's scores, identical
in all three paths, are:

| Metric | Original Ko / ported / registered | Native absolute error |
| --- | --- | --- |
| `prompt_level_strict_acc` | 117/342 (0.34210526) | 0 |
| `inst_level_strict_acc` | 207/481 (0.43035343) | 0 |
| `prompt_level_loose_acc` | 125/342 (0.36549708) | 0 |
| `inst_level_loose_acc` | 220/481 (0.45738046) | 0 |

There were **0 scoring errors, 0 aggregation errors and 0 mismatches**. The full
denominators include every token-limited response. All seven run checksums
passed. An independent tokenizer replay matched every rendered input, input
token sequence, raw decoded output, scoring response, hash and finish reason.
These are saved-response scoring results, with native Lighteval model inference
still outside this validation.

This run used a local working tree based on `9731595bb95455aa442403516114a44a39d740bd`.
The manifest records that base and the uncommitted file list; the exact generator
snapshot is included in the run. The evaluation report records the verifier and
loaded task-source hashes separately. Original local records are preserved.
Publication copies explicitly mark personal execution-path redaction. Inputs,
output token IDs, responses and judgments are unchanged. The public generation
archive passed a fresh three-path replay with the same full denominators, zero
errors and the four scores above.

### Public attachments and checksums

The publication copies and their checksum files are attached to
[PR #7](https://github.com/Hugging-Face-KREW/lighteval/pull/7), rather than checked
into the repository. Download them using the links below:

| Archive and checksum file | SHA-256 |
| --- | --- |
| [qwen-342-001-public.zip](https://github.com/user-attachments/files/34839513/qwen-342-001-public.zip) ([checksum](https://github.com/user-attachments/files/34839521/qwen-342-001-public.zip.sha256.txt)) | `8d398dfbe60656daa4c9cd20bda61a91120fcea86d7e3b9fcebb7c0c91e4ddbb` |
| [qwen-342-parity-001-public.zip](https://github.com/user-attachments/files/34839530/qwen-342-parity-001-public.zip) ([checksum](https://github.com/user-attachments/files/34839550/qwen-342-parity-001-public.zip.sha256.txt)) | `81846baa55d5e64a5090ad80e26efd555727e1e15deb9e510a58d4a01405a368` |

### Reproduce full-split validation

Generate a fresh run with `scripts/ifeval_ko/generate_responses.py --scope full`,
then select the same scope in the saved-response verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 python tests/ifeval_ko/verify_model_responses.py \
  --scope full --run-dir outputs/ifeval_ko/qwen-342-001 \
  --audit-dir "$KO_AUDIT" --source-root "$KO_SOURCES" \
  --output-dir outputs/ifeval_ko/qwen-342-parity-001
```

See the [generation guide](../../scripts/ifeval_ko/README.md#generate-the-full-342-prompt-split)
for the pinned model/configuration and artifact contract. The verifier does not
generate new responses or load model weights. `--scope selected` remains the
default, including support for the original PR #5 manifest without a scope field.
Selected and full artifacts cannot be mixed.

The full scope requires all 342 unique keys in the pinned source split's exact
order, all 481 instruction occurrences and unchanged source fields, including
every occurrence of a repeated instruction ID. Source membership and order are
checked against the hash-verified pinned Parquet, not inferred from the saved
manifest. A self-consistent reordered run is rejected before any scoring call.

All three paths receive each response verbatim, including empty and token-limited
responses. Seeds are reset before each call. Prompt and instruction metrics use
denominators of 342 and 481, respectively. Any missing record, scoring exception,
invalid judgment, aggregation error or unresolved mismatch produces a nonzero
exit status. A scoring exception never produces aggregate scores for a smaller
subset. Reports and failure diagnostics show the requested scope and its full
denominators, even when execution is interrupted.

The full-split negative controls additionally check source order, scope mismatch,
missing/duplicate records, wrong instruction counts, a late generation/log error,
and scoring errors without denominator reduction. The original selected-scope
regressions remain in the same test suites.

## Scoring behavior

The scorer follows Allganize IFEval-Ko's `ifeval_ko/utils.py` at revision
`54199e3801116897697babf341865741dcd06fc8`. It preserves NFC at each phase entry,
original prompt/kwargs, `if v` filtering,
optional second `build_description(prompt=prompt)`, all strict before all loose,
fresh checker per instruction per phase, indexed repeated IDs, eight ordered
loose candidates including duplicates, `strip()` eligibility, per-instruction OR
and per-prompt AND. Empty responses still construct checkers before returning
False. Empty instruction lists retain `all([]) == True`. External checker
exceptions propagate; internally handled exceptions retain the checker's result.

## `main.py` integration contract

```python
from lighteval.tasks.tasks.ifeval_ko.scoring import score_response

# In the Ko SampleLevelComputation.compute(doc, model_response, **kwargs):
return score_response(
    prompt=doc.query,
    instruction_id_list=doc.specific["instructions_id_list"],
    kwargs=doc.specific["kwargs"],
    response=model_response.final_text[0],
)
```

The return keys are `prompt_level_strict_acc` and `prompt_level_loose_acc`
(`bool`), and `inst_level_strict_acc` and `inst_level_loose_acc` (`list[bool]`
in original instruction order). `score_strict` and `score_loose` return
`PhaseResult(follow_instruction_list, follow_all_instructions)`.

Aggregate prompt Booleans by their mean. Flatten all instruction lists before
taking their mean, keeping every repeated ID. Do not average per-prompt
instruction accuracies. For `[[True, True, True], [False]]`, the prompt metric
is 0.5 and the instruction metric is 0.75. The caller preserves the raw prompt
and kwargs and forwards `final_text[0]` without additional scorer-side trimming,
translation, reasoning-tag stripping or Markdown cleanup.

## Reproduction

Read-only audit revision:
`Hugging-Face-KREW/k-llm-eval-sprint@da7053db1d4858239856695a2cfed702173234b6`,
`teams/team-c-ifeval-ko/evidence/checker-audit/`.
Original Ko revision: `54199e3801116897697babf341865741dcd06fc8`.
The verifier checks all `SHA256SUMS` entries, source hashes, dependency versions
and NLTK resource hashes before evaluation. Audit inputs and expected results
are read-only.

Run commands from the Lighteval repository root. If the audit inputs, pinned
sources and dependencies are already available, activate the appropriate Python
environment and supply your own directories. The registered-task verifier needs
an installed Lighteval environment, including its declared optional Ko packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e . 'langdetect==1.0.9' 'immutabledict==4.3.1' \
  'packaging==26.3' 'pyarrow==26.0.0'
```

On WSL, use a Linux-filesystem virtual environment. A CPU build of PyTorch is
sufficient; neither verifier generates model responses.

Run the direct scorer regression using the ported checkers:

```bash
KO_AUDIT=/path/to/checker-audit
KO_SOURCES=/path/to/restored-audit-sources
PYTHONDONTWRITEBYTECODE=1 python3 tests/ifeval_ko/verify_scoring.py \
  --audit-dir "$KO_AUDIT" --source-root "$KO_SOURCES" \
  --checker-mode ported \
  --output results/ifeval_ko/scoring-parity-ported.json
PYTHONDONTWRITEBYTECODE=1 python3 "$KO_AUDIT/test_audit.py"
python3 tests/unit/tasks/test_ifeval_ko_scoring.py
```

Run the real registered task with the same fixtures:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/ifeval_ko/verify_registered_task.py \
  --audit-dir "$KO_AUDIT" --source-root "$KO_SOURCES" \
  --output results/ifeval_ko/registered-task-results.json
```

This independently executes the original Ko evaluator, direct ported scorer,
registered formatter/metric, corpus aggregations and 342-item dataset Doc
conversion. It does not require anyone else's saved results. It records the
actual Git HEAD and hashes of the loaded task sources and verifier.

Optionally compare previously saved direct and registered results as well:

```bash
KO_UPSTREAM_RESULTS=/path/to/registered-task-results.json
PYTHONDONTWRITEBYTECODE=1 python3 tests/ifeval_ko/verify_registered_task.py \
  --audit-dir "$KO_AUDIT" --source-root "$KO_SOURCES" \
  --direct-results results/ifeval_ko/scoring-parity-ported.json \
  --upstream-results "$KO_UPSTREAM_RESULTS" \
  --output results/ifeval_ko/registered-task-comparison.json
```

Saved results must contain per-case decisions, ordered IDs, keys and input/response
hashes, plus the supplemental cases. The registered verifier refuses to overwrite
an existing output; choose new output names when preserving previous reports.

`--audit-dir` identifies the pinned `checker-audit/` evidence directory.
`--source-root` identifies the directory containing `sources/` and
`.cache/nltk_data/`. These may be different directories.

For a fresh environment, set the path to your documentation repository checkout
and export the pinned audit into a separate validation directory. The pinned
commit must be available in that checkout. Setup requires network access:

```bash
KO_DOCS_REPO=/path/to/k-llm-eval-sprint
KO_AUDIT=$(mktemp -d)
git -C "$KO_DOCS_REPO" archive da7053db1d4858239856695a2cfed702173234b6 \
  teams/team-c-ifeval-ko/evidence/checker-audit/ \
  | tar -x -C "$KO_AUDIT" --strip-components=4
python3 -m venv "$KO_AUDIT/.venv"
"$KO_AUDIT/.venv/bin/pip" install -r "$KO_AUDIT/requirements.lock.txt"
"$KO_AUDIT/.venv/bin/python" "$KO_AUDIT/fetch_sources.py"
"$KO_AUDIT/.venv/bin/python" -c \
  'import nltk, sys; assert all(nltk.download(name, download_dir=sys.argv[1], quiet=True) for name in ("punkt", "punkt_tab"))' \
  "$KO_AUDIT/.cache/nltk_data"
PYTHONDONTWRITEBYTECODE=1 "$KO_AUDIT/.venv/bin/python" tests/ifeval_ko/verify_scoring.py \
  --audit-dir "$KO_AUDIT" --source-root "$KO_AUDIT" --checker-mode ported
```

Preload both NLTK resources during setup. Loading the registered task also imports
the existing English IFEval utilities, which attempt to download missing `punkt`
resources. Preloading them keeps the scoring run from writing to the source cache.

The existing baseline `test_audit.py` validates the reference evidence, including
negative controls. It does not execute the new scorer. `verify_scoring.py`
separately invokes `score_response` on all 156 full cases, matching by `case_id`.
It uses `ko_eval_all_kwargs` when present and otherwise `ko_all_kwargs`.
Each case is seeded with Python `random.seed(0)`; `DetectorFactory.seed=0`;
one worker is used. The verifier also compares actual candidate order, checker
creation/build/call traces and input immutability against the archived reference.
Supplementary strict and loose entry points are each checked, including when
strict raises; the combined entry's exception propagation is checked separately.

## Environment and result artifact

Validated with Python 3.12.3; nltk 3.10.3; langdetect 1.0.9; immutabledict 4.3.1;
packaging 26.3; pyarrow 26.0.0. The registered-task run used CPU PyTorch
2.14.1+cpu, transformers 5.19.0 and inspect-ai 0.3.280 in a separate Linux
environment. The result records its package versions. Numeric-text language detection logs from `123`
are emitted by the original checker, which internally catches the exception
and returns True; these are not unhandled scoring failures.

`scoring-parity-ported.json` and `registered-task-results.json` contain the actual
command/environment, source/evidence/verifier hashes, each case's flags, metric
counts, auxiliary outcomes and mismatch list. They are local artifacts ignored by
the repository's existing `*.json` rule. Keep these reports as PR attachments
rather than committing machine-specific runtime paths. On failure, the report includes
the first mismatch's original input and old/new judgments; instruction-level
differences include `key`, `instruction_index`, and `instruction_id`.

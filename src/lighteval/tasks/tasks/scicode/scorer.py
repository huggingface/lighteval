"""Scorer and metrics for SciCode evaluation.

Based on original implementation:
https://github.com/scicode-bench/SciCode
"""

import platform
import resource
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from inspect_ai.scorer import Metric, Score, Target, mean, metric, scorer
from inspect_ai.solver import TaskState

from lighteval.tasks.tasks.lcb.codegen_metrics import extract_code
from lighteval.tasks.tasks.scicode.parse import extract_targets
from lighteval.tasks.tasks.scicode.solver import should_skip_step
from lighteval.tasks.tasks.scicode.utils import get_h5py_file_path


@metric
def sub_problem_correctness() -> Metric:
    """Metric to compute sub-problem correctness rate."""

    def metric_fn(scores: list[Score]) -> int | float:
        total_correct = 0
        total_steps = 0
        for score in scores:
            total_correct += score.value["Total Correct"]
            total_steps += score.value["Total Steps"]
        return total_correct / total_steps if total_steps > 0 else 0.0

    return metric_fn


def run_script(script_path: Path) -> int:
    """Run test script and return exit code.

    0 = pass, 1 = fail, 2 = timeout

    Note: Resource limits are applied to restrict memory usage (4GB max).
    """
    maximum_memory_bytes = 4 * 1024 * 1024 * 1024  # 4GB

    def set_resource_limits():
        """Set resource limits in the child process before execution."""
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.system() != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    preexec_fn = set_resource_limits if platform.system() != "Windows" else None

    process = None
    try:
        process = subprocess.Popen(
            ["python", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec_fn,
        )
        stdout, stderr = process.communicate(timeout=1800)  # 30 minutes like original
        if process.returncode == 0:
            return 0
        return 1
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
            process.wait()
        return 2
    except Exception:
        return 1


def _get_initial_score(sub_steps: list) -> Score | None:
    """Check for early return conditions and return appropriate Score if needed.

    Returns Score for early returns, or None if processing should continue.
    """
    if not sub_steps:
        return Score(
            value={
                "Problem Correctness": 0,
                "Total Correct": 0,
                "Total Steps": 0,
            },
            explanation="No sub-steps found in metadata",
        )
    return None


def _get_h5py_file_or_error_score(sub_steps: list) -> tuple[Path, Score | None]:
    """Get h5py file path or return error Score if it fails.

    Returns tuple of (h5py_file, error_score) where error_score is None on success.
    """
    try:
        h5py_file = get_h5py_file_path()
        return h5py_file, None
    except Exception as e:
        error_score = Score(
            value={
                "Problem Correctness": 0,
                "Total Correct": 0,
                "Total Steps": len(sub_steps),
            },
            explanation=f"Failed to get h5py file: {e}",
        )
        return None, error_score


def _get_code_content(step_id: str, generated_code_by_step: dict, state: TaskState) -> str | None:
    """Get code content for a step, with fallback to state output."""
    code_content = generated_code_by_step.get(step_id, "")
    if not code_content:
        response = state.output.completion if hasattr(state, "output") else ""
        code_content = extract_code(response) if response else ""
    return code_content if code_content else None


def _write_test_script(test_script: Path, code_content: str, targets: tuple, test_cases: list[str]) -> None:
    """Write test script file with imports, code, targets, and test cases."""
    with open(test_script, "w", encoding="utf-8") as f:
        f.write("import numpy as np\n")
        f.write("from numpy import array\n\n")
        f.write(code_content)
        f.write("\n\n")
        f.write("targets = (\n")
        for target in targets:
            if hasattr(target, "tolist"):
                f.write(f"    np.array({target.tolist()}),\n")
            elif hasattr(target, "__iter__") and not isinstance(target, str):
                f.write(f"    {repr(target)},\n")
            else:
                f.write(f"    {repr(target)},\n")
        f.write(")\n\n")
        for i in range(len(test_cases)):
            f.write(f"target = targets[{i}]\n\n")
            f.write(test_cases[i])
            f.write("\n\n")


def _execute_and_aggregate_tests(sub_steps: list, problem_id: str, tmp_dir: Path) -> tuple[int, int]:
    """Execute test scripts and aggregate results.

    Returns tuple of (total_correct, total_steps).
    """
    total_correct = 0
    total_steps = 0

    for idx in range(len(sub_steps)):
        if should_skip_step(problem_id, idx):
            continue

        step_data = sub_steps[idx]
        step_id = step_data.get("step_number")

        if not step_id:
            continue

        total_steps += 1
        script_path = tmp_dir / f"{step_id}.py"

        if not script_path.exists():
            continue

        ret = run_script(script_path)
        if ret == 0:
            total_correct += 1

    return total_correct, total_steps


@scorer(
    metrics=[
        {"Problem Correctness": [mean()]},
        sub_problem_correctness(),
    ]
)
def scicode_scorer():
    """Scorer for SciCode evaluation using inspect-ai.

    Implements full multi-step test execution with h5py file support.
    """

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata
        sub_steps = metadata.get("sub_steps", [])
        problem_id = metadata.get("problem_id")
        generated_code_by_step = metadata.get("generated_code_by_step", {})

        initial_score = _get_initial_score(sub_steps)
        if initial_score is not None:
            return initial_score

        h5py_file, error_score = _get_h5py_file_or_error_score(sub_steps)
        if error_score is not None:
            return error_score

        tmp_dir: Path | None = None
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"scicode_test_{uuid.uuid4().hex}_"))
            for idx in range(len(sub_steps)):
                if should_skip_step(problem_id, idx):
                    continue

                step_data = sub_steps[idx]
                step_id = step_data.get("step_number")
                test_cases = step_data.get("test_cases", [])

                if not step_id or not test_cases:
                    continue

                code_content = _get_code_content(step_id, generated_code_by_step, state)
                if not code_content:
                    continue

                try:
                    targets = extract_targets(step_id, len(test_cases), h5py_file)
                except Exception:
                    continue

                test_script = tmp_dir / f"{step_id}.py"
                _write_test_script(test_script, code_content, targets, test_cases)

            total_correct, total_steps = _execute_and_aggregate_tests(sub_steps, problem_id, tmp_dir)

            problem_correct = 1 if total_correct == total_steps and total_steps > 0 else 0

            return Score(
                value={
                    "Problem Correctness": problem_correct,
                    "Total Correct": total_correct,
                    "Total Steps": total_steps,
                },
                explanation=f"Tested {total_steps} steps, {total_correct} passed",
            )

        finally:
            if tmp_dir is not None and tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    return score

"""
name:
SciCode

dataset:
SciCode1/SciCode

abstract:
SciCode is a challenging benchmark designed to evaluate the capabilities of language models (LMs)
in generating code for solving realistic scientific research problems. It has a diverse coverage of
16 subdomains from 6 domains: Physics, Math, Material Science, Biology, and Chemistry. Unlike previous
benchmarks that consist of exam-like question-answer pairs, SciCode is converted from real research problems.
SciCode problems naturally factorize into multiple subproblems, each involving knowledge recall, reasoning,
and code synthesis. In total, SciCode contains 338 subproblems decomposed from 80 challenging main problems.

languages:
english

tags:
code-generation, scientific-computing

paper:
https://arxiv.org/abs/2407.13168

starred:
true
"""

from typing import Any

from inspect_ai.dataset import Sample

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.scicode.prompts import prepare_scicode_prompt
from lighteval.tasks.tasks.scicode.scorer import scicode_scorer
from lighteval.tasks.tasks.scicode.solver import scicode_solver
from lighteval.tasks.tasks.scicode.utils import _extract_first_step_metadata


def scicode_prompt(line: dict[str, Any], task_name: str = "scicode") -> Doc:
    """Convert dataset record to Doc for evaluation.

    For multi-step evaluation, this returns the first step's prompt.
    The solver will handle subsequent steps.
    """
    step_metadata = _extract_first_step_metadata(line)
    step_data = step_metadata["step_data"]
    query = prepare_scicode_prompt(step_data, line, with_background=False)

    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "test_cases": step_metadata["test_cases"],
            "function_header": step_metadata["function_header"],
            "fn_name": step_metadata["fn_name"],
            "step_number": step_metadata["step_number"],
            "problem_id": line.get("problem_id"),
            "required_dependencies": line.get("required_dependencies", ""),
        },
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert dataset record to inspect-ai Sample object.

    Includes ALL sub_steps in metadata for multi-step processing.
    """
    step_metadata = _extract_first_step_metadata(record)
    step_data = step_metadata["step_data"]

    metadata = dict(record)
    metadata.update(
        {
            "test_cases": step_metadata["test_cases"],
            "function_header": step_metadata["function_header"],
            "fn_name": step_metadata["fn_name"],
            "step_number": step_metadata["step_number"],
        }
    )

    prompt = prepare_scicode_prompt(step_data, record, with_background=False)

    return Sample(input=prompt, metadata=metadata)


scicode = LightevalTaskConfig(
    name="scicode",
    prompt_function=scicode_prompt,
    sample_fields=record_to_sample,
    solver=scicode_solver(with_background=False),
    scorer=scicode_scorer(),
    hf_repo="SciCode1/SciCode",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    generation_size=32768,
    metrics=[],  # Metrics are defined in the scorer decorator for inspect_ai
    stop_sequence=[],  # no stop sequence, will use EOS token
    version=0,
)

TASKS_TABLE = [scicode]

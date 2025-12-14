"""Multi-step solver for SciCode.

Based on original implementation:
https://github.com/scicode-bench/SciCode
"""

import copy
from typing import Any

from inspect_ai.solver import Generate, TaskState, solver

from lighteval.tasks.tasks.scicode.prompts import SCICODE_PROMPT_TEMPLATE, generate_prompt_with_steps
from lighteval.tasks.tasks.scicode.utils import extract_python_script


def should_skip_step(problem_id: str, step_idx: int) -> bool:
    """Check if a step should be skipped based on special cases.

    Special cases from original implementation:
    - Problem 13, step 6 (idx 5)
    - Problem 62, step 1 (idx 0)
    - Problem 76, step 3 (idx 2)
    """
    return (
        (problem_id == "13" and step_idx == 5)
        or (problem_id == "62" and step_idx == 0)
        or (problem_id == "76" and step_idx == 2)
    )


@solver
def scicode_solver(**params: dict[str, Any]):
    """Custom solver that processes all sub-steps sequentially."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        sub_steps = state.metadata.get("sub_steps", [])
        problem_id = state.metadata.get("problem_id")
        with_background = params.get("with_background", False)

        if not sub_steps:
            return state

        if "generated_code_by_step" not in state.metadata:
            state.metadata["generated_code_by_step"] = {}

        tot_steps = len(sub_steps)
        previous_llm_code: list[str | None] = [None] * tot_steps

        for idx in range(len(sub_steps)):
            if should_skip_step(problem_id, idx):
                continue

            num_steps = idx + 1

            prompt, previous_code_str = generate_prompt_with_steps(
                prob_data=state.metadata,
                num_steps=num_steps,
                previous_llm_code=previous_llm_code,
                prompt_template=SCICODE_PROMPT_TEMPLATE,
                with_background=with_background,
            )

            try:
                state.user_prompt.text = prompt
                state_copy = copy.deepcopy(state)
                result = await generate_fn(state=state_copy)
                response_from_llm = result.output.completion
            except Exception:
                return state

            extracted_code = extract_python_script(response_from_llm)
            step_id = sub_steps[idx].get("step_number")

            if step_id:
                state.metadata["generated_code_by_step"][step_id] = extracted_code
                previous_llm_code[idx] = extracted_code

        return state

    return solve

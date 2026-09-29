"""Prompt templates and generation for SciCode."""

from typing import Any


SCICODE_PROMPT_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{previous_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```""".strip()


def prepare_scicode_prompt(
    step_data: dict[str, Any], problem_data: dict[str, Any], with_background: bool = False
) -> str:
    """Prepare prompt for the first SciCode sub-step (no previous steps).

    This function is used for initial prompt generation before the solver runs.
    For subsequent steps with previous context, use generate_prompt_with_steps() instead.
    """
    next_step_parts = [step_data["step_description_prompt"]]

    if with_background and step_data.get("step_background"):
        next_step_parts.append(step_data["step_background"])

    next_step_parts.append(step_data["function_header"])

    if step_data.get("return_line"):
        next_step_parts.append(step_data["return_line"])

    next_step_str = "\n\n".join(next_step_parts)
    dependencies = problem_data.get("required_dependencies", "")
    previous_steps_str = ""

    prompt = SCICODE_PROMPT_TEMPLATE.format(
        previous_steps_str=previous_steps_str,
        next_step_str=next_step_str,
        dependencies=dependencies,
    )

    return prompt


def process_problem_code(prob_data: dict[str, Any], num_steps: int) -> str:
    """Extract function header and return line for a given step."""
    header_docstring = prob_data["sub_steps"][num_steps - 1]["function_header"]
    return_str = prob_data["sub_steps"][num_steps - 1].get("return_line", "")
    if return_str:
        return f"{header_docstring}\n\n{return_str}"
    return header_docstring


def process_problem_steps(
    problem_data: dict[str, Any],
    num_steps: int,
    previous_llm_code: list[str | None],
    with_background: bool = False,
) -> tuple[str, str, str]:
    """Process problem data and return previous steps and next steps.

    Returns:
        tuple: (previous_steps_str, next_step_str, previous_code_str)
    """
    output_lines = []
    next_step = []
    previous_code = []

    for i in range(num_steps - 1):
        step_desc = problem_data["sub_steps"][i]["step_description_prompt"]
        if with_background and problem_data["sub_steps"][i].get("step_background"):
            step_desc += "\n" + problem_data["sub_steps"][i]["step_background"]
        output_lines.append(step_desc)

        if previous_llm_code[i] is not None:
            output_lines.append(previous_llm_code[i])
            previous_code.append(previous_llm_code[i])
        output_lines.append("------")

    # Next step
    step_desc = problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
    if with_background and problem_data["sub_steps"][num_steps - 1].get("step_background"):
        step_desc += "\n" + problem_data["sub_steps"][num_steps - 1]["step_background"]
    next_step.append(step_desc)
    next_step.append(process_problem_code(problem_data, num_steps))

    output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)

    return output_str, next_step_str, previous_code_str


def generate_prompt_with_steps(
    prob_data: dict[str, Any],
    num_steps: int,
    previous_llm_code: list[str | None],
    prompt_template: str = SCICODE_PROMPT_TEMPLATE,
    with_background: bool = False,
) -> tuple[str, str]:
    """Generate prompt for step N with previous steps context.

    Returns:
        tuple: (prompt, previous_code_str)
    """
    problem_steps_str, next_step_str, previous_code_str = process_problem_steps(
        prob_data, num_steps, previous_llm_code, with_background
    )
    dependencies = prob_data.get("required_dependencies", "")

    prompt = prompt_template.format(
        previous_steps_str=problem_steps_str,
        next_step_str=next_step_str,
        dependencies=dependencies,
    )

    previous_code_with_deps = f"{dependencies}\n{previous_code_str}\n" if previous_code_str else f"{dependencies}\n"

    return prompt, previous_code_with_deps

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

import re
from typing import Any

import numpy as np
from aenum import extend_enum
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.lcb.codegen_metrics import extract_code


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


def extract_function_name(function_header: str) -> str:
    """Extract function or class name from function header."""
    pattern = r"\bdef\s+(\w+)\s*\("
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)

    pattern = r"\bclass\s+(\w+)\s*[\(:]"
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)

    raise ValueError(f"Function name or class name not found in: {function_header}")


def prepare_scicode_prompt(
    step_data: dict[str, Any], problem_data: dict[str, Any], with_background: bool = False
) -> str:
    """Prepare prompt for a SciCode sub-step following the official template structure."""
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


def scicode_prompt(line: dict[str, Any], task_name: str = "scicode") -> Doc:
    """Convert dataset record to Doc for evaluation."""
    if not line.get("sub_steps") or len(line["sub_steps"]) == 0:
        raise ValueError("No sub-steps found in problem data")

    step_data = line["sub_steps"][0]
    query = prepare_scicode_prompt(step_data, line, with_background=False)

    test_cases = step_data.get("test_cases", [])
    function_header = step_data.get("function_header", "")
    fn_name = extract_function_name(function_header) if function_header else None

    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "test_cases": test_cases,
            "function_header": function_header,
            "fn_name": fn_name,
            "step_number": step_data.get("step_number"),
            "problem_id": line.get("problem_id"),
            "required_dependencies": line.get("required_dependencies", ""),
        },
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert dataset record to inspect-ai Sample object."""
    if not record.get("sub_steps") or len(record["sub_steps"]) == 0:
        raise ValueError("No sub-steps found in problem data")

    step_data = record["sub_steps"][0]
    function_header = step_data.get("function_header", "")
    fn_name = extract_function_name(function_header) if function_header else None

    metadata = {
        "test_cases": step_data.get("test_cases", []),
        "function_header": function_header,
        "fn_name": fn_name,
        "step_number": step_data.get("step_number"),
        "problem_id": record.get("problem_id"),
        "required_dependencies": record.get("required_dependencies", ""),
        "problem_description": record.get("problem_description_main", ""),
        "step_description": step_data.get("step_description_prompt", ""),
    }

    prompt = prepare_scicode_prompt(step_data, record, with_background=False)

    return Sample(input=prompt, metadata=metadata)


class SciCodeMetric(SampleLevelComputation):
    """Metric for SciCode code generation evaluation."""

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        """Check if code was generated."""
        assert doc.specific is not None, "Doc specific field is required for scicode metric"

        predictions = model_response.final_text

        if not predictions:
            return {"code_extracted": 0.0}

        generated_code = extract_code(predictions[0])
        code_extracted = 1.0 if generated_code and len(generated_code.strip()) > 0 else 0.0

        return {"code_extracted": code_extracted}


scicode_metric = SampleLevelMetric(
    metric_name="scicode_code_extracted",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=SciCodeMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

extend_enum(Metrics, "scicode_metric", scicode_metric)


@scorer(metrics=[accuracy(), stderr()])
def scicode_scorer():
    """Scorer for SciCode evaluation using inspect-ai."""

    async def score(state: TaskState, target: Target):
        response = state.output.completion
        generated_code = extract_code(response)

        if not generated_code or len(generated_code.strip()) == 0:
            return Score(value="I", explanation="No code found in response", answer="")

        return Score(
            value="C",
            explanation="Code successfully extracted from response",
            answer=generated_code[:200],
        )

    return score


scicode = LightevalTaskConfig(
    name="scicode",
    prompt_function=scicode_prompt,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=scicode_scorer(),
    hf_repo="SciCode1/SciCode",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    generation_size=32768,
    metrics=[Metrics.scicode_metric],
    stop_sequence=[],  # no stop sequence, will use EOS token
    version=0,
)

TASKS_TABLE = [scicode]

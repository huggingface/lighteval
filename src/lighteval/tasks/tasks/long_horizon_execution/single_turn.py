"""
Single turn implementation of the Long Horizon Execution task.
"""

import functools
import re

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.long_horizon_execution.constants import CONTEXT_SIZES, PROMPT_TEMPLATE_SINGLE
from lighteval.tasks.tasks.long_horizon_execution.utils import _build_prompt_and_target


def single_turn_prompt_function(line, prompt_length=32768, task_name: str = None):
    """
    Prompt function for single-turn evaluation (non-inspect-ai backend).
    Converts dataset record to Doc object.
    Returns:
        Doc object for evaluation
    """
    prompt, target_str, _ = _build_prompt_and_target(
        line, prompt_length=prompt_length, prompt_template=PROMPT_TEMPLATE_SINGLE
    )

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[target_str],  # Expected answer as a choice
        gold_index=0,
        instruction=prompt,
    )


def single_turn_record_to_sample(record, prompt_length=32768):
    """
    Converts dataset record to inspect-ai Sample object for single-turn evaluation.
    Returns:
        Sample object for inspect-ai
    """
    prompt, target_str, metadata = _build_prompt_and_target(
        record, prompt_length=prompt_length, prompt_template=PROMPT_TEMPLATE_SINGLE
    )

    return Sample(
        input=prompt,
        target=target_str,
        metadata=metadata,
    )


@scorer(metrics=[accuracy(), stderr()])
def single_turn_scorer():
    """
    Scorer for single-turn evaluation.
    Compares the model's predicted final sum with the expected final sum (binary score).
    Returns:
        Scorer function that evaluates single integer responses
    """

    async def score(state: TaskState, target: Target):
        response = state.output.completion

        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        match = answer_pattern.search(response)

        if not match:
            return Score(value="I", answer="", explanation="No <answer> tag found in response.")

        content = match.group(1).strip()

        try:
            pred_value = int(content.strip())
        except ValueError:
            return Score(value="I", answer=content, explanation=f"Failed to parse integer from: {content}")

        try:
            exp_value = int(target.text.strip())
        except (ValueError, AttributeError):
            return Score(
                value="I",
                answer=str(pred_value),
                explanation=f"Failed to parse expected target: {target.text}",
            )

        is_correct = pred_value == exp_value
        return Score(
            value="C" if is_correct else "I",
            answer=str(pred_value),
            explanation=(f"Expected {exp_value}, Got {pred_value}. Match: {is_correct}"),
        )

    return score


def create_single_turn_tasks():
    """
    Create all single-turn task configurations for different context sizes.
    Returns:
        list[LightevalTaskConfig]: List of task configurations for single-turn evaluation
    """
    tasks = []

    for context_size in CONTEXT_SIZES:
        task_name = f"long_horizon_execution:{context_size}"
        prompt_fn = functools.partial(single_turn_prompt_function, prompt_length=context_size)
        sample_fn = functools.partial(single_turn_record_to_sample, prompt_length=context_size)

        task = LightevalTaskConfig(
            name=task_name,
            prompt_function=prompt_fn,
            sample_fields=sample_fn,
            solver=[generate(cache=True)],
            scorer=single_turn_scorer(),
            hf_repo="arvindh75/Long-Horizon-Execution",
            hf_subset="default",
            evaluation_splits=("test",),
            generation_size=context_size,
            metrics=[Metrics.exact_match],
        )

        tasks.append(task)

    return tasks

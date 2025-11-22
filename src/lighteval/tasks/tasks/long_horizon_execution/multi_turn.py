"""
Multi-turn implementation of the Long Horizon Execution task.
This implementation matches the multi-turn evaluation approach from the research paper,
where keys are provided in batches of K per turn, and the model maintains conversation
state to output cumulative sums after each turn.
"""

import functools
import re

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.long_horizon_execution.constants import (
    CONTEXT_SIZES,
    PROMPT_TEMPLATE_MULTI_FOLLOWUP,
    TURN_COMPLEXITIES,
)
from lighteval.tasks.tasks.long_horizon_execution.utils import _build_multi_turn_prompts


def multi_turn_prompt_function(line, prompt_length=32768, k=1, task_name: str = None):
    """
    Prompt function for non-inspect-ai backend for multi-turn evaluation.
    Converts dataset record to Doc object.
    Note: For multi-turn, this returns the first turn's prompt.
    Subsequent turns are handled by the solver.
    """
    initial_prompt, _, expected_per_turn, _ = _build_multi_turn_prompts(line, prompt_length=prompt_length, k=k)

    return Doc(
        task_name=task_name,
        query=initial_prompt,
        choices=[str(expected_per_turn[-1])],  # Final sum as choice
        gold_index=0,
        instruction=initial_prompt,
    )


def multi_turn_record_to_sample(record, prompt_length=32768, k=1):
    """
    Converts dataset record to inspect-ai Sample object for multi-turn evaluation.
    Stores all turn information in metadata for the solver to use.
    """
    initial_prompt, _, expected_per_turn, metadata = _build_multi_turn_prompts(
        record, prompt_length=prompt_length, k=k
    )

    return Sample(
        input=initial_prompt,
        target=str(expected_per_turn[-1]),
        metadata=metadata,
    )


def _extract_response_content(response):
    """Extract content from model response object."""
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "completion"):
        return response.completion
    return str(response)


async def _process_single_turn(state, turn_chunk, config):
    """Process a single turn: add user message, get model response, add assistant message."""
    keys_str = ", ".join(turn_chunk)
    followup_prompt = PROMPT_TEMPLATE_MULTI_FOLLOWUP.format(keys_str=keys_str)
    state.messages.append({"role": "user", "content": followup_prompt})

    response = await state.model.generate(messages=state.messages, config=config)
    turn_response = _extract_response_content(response)

    state.messages.append({"role": "assistant", "content": turn_response})
    return turn_response


async def multi_turn_solver(state: TaskState):
    """
    Custom solver for multi-turn evaluation.
    Loops through turns, calling the model for each turn while maintaining conversation history.
    This implements offline evaluation: all turns are called, then evaluation happens.
    """
    from inspect_ai.model import GenerateConfig, ModelOutput

    turn_chunks = state.metadata.get("turn_chunks", [])

    if not turn_chunks or len(turn_chunks) == 0:
        return state

    # Initialize messages
    if not hasattr(state, "messages") or state.messages is None:
        state.messages = []

    if not state.messages:
        state.messages.append({"role": "user", "content": state.input})

    all_turn_outputs = []

    # Process all turns
    if hasattr(state, "model") and state.model is not None:
        config = GenerateConfig()

        # Process first turn (already in messages as initial prompt)
        response = await state.model.generate(messages=state.messages, config=config)
        turn_response = _extract_response_content(response)
        all_turn_outputs.append(turn_response)
        state.messages.append({"role": "assistant", "content": turn_response})

        # Process remaining turns
        for turn_idx in range(1, len(turn_chunks)):
            if not hasattr(state, "model") or state.model is None:
                break
            turn_response = await _process_single_turn(state, turn_chunks[turn_idx], config)
            all_turn_outputs.append(turn_response)

    state.metadata["all_turn_outputs"] = all_turn_outputs

    # Set final output
    if all_turn_outputs:
        if hasattr(state, "output") and state.output is not None:
            state.output.completion = all_turn_outputs[-1]
        else:
            state.output = ModelOutput(completion=all_turn_outputs[-1])

    return state


@scorer(metrics={"turn_accuracy": [accuracy(), stderr()], "fractional_accuracy": [accuracy(), stderr()]})
def multi_turn_scorer():
    """
    Scorer for multi-turn Long Horizon Execution task.
    Compares predicted cumulative sums at each turn with expected.
    Returns fractional accuracy (correct turns / total turns).
    """

    async def score(state: TaskState, target: Target):
        # metadata stored by solver
        all_turn_outputs = state.metadata.get("all_turn_outputs", [])
        expected_per_turn = state.metadata.get("expected_per_turn", [])

        if not all_turn_outputs:
            return Score(value=0.0, answer="", explanation="No turn outputs found in state.metadata")

        if len(all_turn_outputs) != len(expected_per_turn):
            return Score(
                value=0.0,
                answer="",
                explanation=f"Mismatch: {len(all_turn_outputs)} outputs vs {len(expected_per_turn)} expected turns",
            )

        parsed_outputs = []
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        for turn_idx, turn_output in enumerate(all_turn_outputs):
            match = answer_pattern.search(turn_output)
            if match:
                try:
                    parsed_value = int(match.group(1).strip())
                    parsed_outputs.append(parsed_value)
                except ValueError:
                    parsed_outputs.append(None)
            else:
                parsed_outputs.append(None)

        correct_turns = 0
        turn_results = []
        for turn_idx, (pred, exp) in enumerate(zip(parsed_outputs, expected_per_turn)):
            is_correct = (pred is not None) and (pred == exp)
            if is_correct:
                correct_turns += 1
            turn_results.append({"turn": turn_idx + 1, "predicted": pred, "expected": exp, "correct": is_correct})

        fractional_accuracy = correct_turns / len(expected_per_turn) if expected_per_turn else 0.0

        return Score(
            value={
                "turn_accuracy": fractional_accuracy,
                "fractional_accuracy": fractional_accuracy,
                "correct_turns": correct_turns,
                "total_turns": len(expected_per_turn),
            },
            answer=str(parsed_outputs),
            explanation=f"Correct {correct_turns}/{len(expected_per_turn)} turns. Details: {turn_results}",
        )

    return score


def create_multi_turn_tasks():
    """
    Creates a list of LightevalTaskConfig objects for multi-turn Long Horizon Execution.
    Each task corresponds to a different combination of context size and turn complexity (K).
    """
    tasks = []

    for context_size in CONTEXT_SIZES:
        for k in TURN_COMPLEXITIES:
            task_name = f"long_horizon_execution:multi:{context_size}:k{k}"
            prompt_fn = functools.partial(multi_turn_prompt_function, prompt_length=context_size, k=k)
            sample_fn = functools.partial(multi_turn_record_to_sample, prompt_length=context_size, k=k)

            task = LightevalTaskConfig(
                name=task_name,
                prompt_function=prompt_fn,
                sample_fields=sample_fn,
                solver=[multi_turn_solver, generate(cache=True)],
                scorer=multi_turn_scorer(),
                hf_repo="arvindh75/Long-Horizon-Execution",
                hf_subset="default",
                evaluation_splits=("test",),
                generation_size=context_size,
                metrics=[Metrics.exact_match],
            )
            tasks.append(task)

    return tasks

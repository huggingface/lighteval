"""
Multi-turn implementation of the Long Horizon Execution task.
This implementation matches the multi-turn evaluation approach from the research paper,
where keys are provided in batches of K per turn, and the model maintains conversation
state to output cumulative sums after each turn.
"""

import functools
import re

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, generate, solver

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


async def _process_single_turn(state, turn_chunk, generate_fn):
    """Process a single turn: add user message, get model response, add assistant message."""
    keys_str = ", ".join(turn_chunk)
    followup_prompt = PROMPT_TEMPLATE_MULTI_FOLLOWUP.format(keys_str=keys_str)
    state.messages.append(ChatMessageUser(content=followup_prompt))

    # generate_fn() takes the state and returns updated state with assistant message added
    updated_state = await generate_fn(state)
    turn_response = _extract_response_content(updated_state.output.completion if updated_state.output else "")

    return updated_state, turn_response


@solver
def multi_turn_solver():
    """
    Solver for multi-turn evaluation.
    Loops through turns, calling the model for each turn while maintaining conversation history.
    This implements offline evaluation: all turns are called, then evaluation happens.
    """

    async def solve(state: TaskState, generate: Generate):
        turn_chunks = state.metadata.get("turn_chunks", [])

        if not turn_chunks:
            return state

        # Initialize messages
        if not hasattr(state, "messages") or state.messages is None:
            state.messages = []

        if not state.messages:
            state.messages.append(ChatMessageUser(content=state.input))

        all_turn_outputs = []

        # Process first turn (already in messages as initial prompt)
        updated_state = await generate(state)
        turn_response = _extract_response_content(updated_state.output.completion if updated_state.output else "")
        all_turn_outputs.append(turn_response)

        state = updated_state

        # Process remaining turns
        for turn_idx in range(1, len(turn_chunks)):
            state, turn_response = await _process_single_turn(state, turn_chunks[turn_idx], generate)
            all_turn_outputs.append(turn_response)

        state.metadata["all_turn_outputs"] = all_turn_outputs

        # Set final output
        if all_turn_outputs:
            if hasattr(state, "output") and state.output is not None:
                state.output.completion = all_turn_outputs[-1]
            else:
                state.output = ModelOutput(completion=all_turn_outputs[-1])

        return state

    return solve


@scorer(metrics={"fractional_accuracy": [accuracy(), stderr()]})
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
            return Score(
                value={"fractional_accuracy": 0.0},
                answer="",
                explanation="No turn outputs found in state.metadata",
            )

        if len(all_turn_outputs) != len(expected_per_turn):
            return Score(
                value={"fractional_accuracy": 0.0},
                answer="",
                explanation=f"Mismatch: {len(all_turn_outputs)} outputs vs {len(expected_per_turn)} expected turns",
            )

        parsed_outputs = []
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        for turn_output in all_turn_outputs:
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
            value={"fractional_accuracy": fractional_accuracy},
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
            task_name = f"long_horizon_execution_multi_k{k}:{context_size}"
            prompt_fn = functools.partial(multi_turn_prompt_function, prompt_length=context_size, k=k)
            sample_fn = functools.partial(multi_turn_record_to_sample, prompt_length=context_size, k=k)

            task = LightevalTaskConfig(
                name=task_name,
                prompt_function=prompt_fn,
                sample_fields=sample_fn,
                solver=[multi_turn_solver(), generate(cache=True)],
                scorer=multi_turn_scorer(),
                hf_repo="arvindh75/Long-Horizon-Execution",
                hf_subset="default",
                evaluation_splits=("test",),
                generation_size=context_size,
                metrics=[Metrics.exact_match],
            )
            tasks.append(task)

    return tasks

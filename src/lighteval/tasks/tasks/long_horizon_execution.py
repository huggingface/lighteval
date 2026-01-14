"""
name:
Long Horizon Execution

dataset:
arvindh75/Long-Horizon-Execution

abstract:
Evaluation benchmark for long-context execution capabilities of language models.
Tests a model's ability to maintain state and perform cumulative operations over
long sequences of inputs. Supports both single-turn (all inputs at once) and
multi-turn (inputs provided incrementally) evaluation modes.
The task requires models to:
1. Maintain a dictionary mapping keys to values
2. Process a sequence of keys
3. Calculate cumulative sums after each key or group of keys
4. Handle varying context sizes and turn complexities
Single-turn evaluation (Section 3.3): Model outputs only the final cumulative sum
after processing all keys, allowing any aggregation strategy.

Multi-turn evaluation: Model processes keys in batches of K per turn, maintaining
conversation history and outputting cumulative sums incrementally. Evaluates
fractional accuracy (correct turns / total turns).

languages:
english

tags:
long-context, state-tracking, arithmetic, execution

paper:
https://arxiv.org/abs/2509.09677

starred:
true
"""

import functools
import itertools
import re

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


PROMPT_TEMPLATE_MULTI_FOLLOWUP = """
I will now provide you with the next {k} keys to process:

{keys_str}
""".strip()

PROMPT_TEMPLATE_MULTI_START = """
I will provide you with a dictionary and then give you the first {k} keys to process.
Your task is to keep a running total (starting from 0) by adding the values associated with the keys I provide.
In each turn, I'll provide {k} keys (comma-separated).
Respond with the current running sum, enclosed in <answer> tags.

Dictionary to maintain:
{dict_str}

Ready to start!

{keys_str}
""".strip()


def record_to_sample(record, k=1, max_turns=5):
    input_keys, input_values = record["input"], record["values"]

    dictionary = dict(zip(input_keys, input_values))
    dictionary_str = str(dictionary)

    keys_per_turn = [input_keys[i : i + k] for i in range(0, len(input_keys), k)][:max_turns]
    values_per_turn = [input_values[i : i + k] for i in range(0, len(input_values), k)][:max_turns]

    targets_per_turn = list(itertools.accumulate(sum(values) for values in values_per_turn))

    initial_prompt = PROMPT_TEMPLATE_MULTI_START.format(dict_str=dictionary_str, keys_str=str(keys_per_turn[0]), k=k)

    metadata = {
        "keys_per_turn": keys_per_turn,
        "values_per_turn": values_per_turn,
        "targets_per_turn": targets_per_turn,
        "k": k,
        "max_turns": max_turns,
    }

    return Sample(
        input=initial_prompt,
        target=str(targets_per_turn[-1]),  # last turn cumulative sum
        metadata=metadata,
    )


@solver
def solver():
    async def solve(state: TaskState, generate: Generate):
        keys_per_turn = state.metadata["keys_per_turn"]

        all_turn_outputs = []

        # Process first turn (already in messages as initial prompt)
        state = await generate(state)
        all_turn_outputs.append(state.output.completion)

        # Process remaining turns
        for keys in keys_per_turn[1:]:
            keys_str = ", ".join(keys)
            followup_prompt = PROMPT_TEMPLATE_MULTI_FOLLOWUP.format(keys_str=keys_str, k=state.metadata["k"])
            state.messages.append(ChatMessageUser(content=followup_prompt))
            state = await generate(state)
            all_turn_outputs.append(state.output.completion)

        state.metadata["all_turn_outputs"] = all_turn_outputs

        return state

    return solve


@scorer(metrics={"horizon": [mean()], "turn_accuracy": [mean()], "all_correct": [accuracy()]})
def scorer():
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    async def score(state: TaskState, target: Target):
        all_turn_outputs = state.metadata.get("all_turn_outputs", [])
        targets_per_turn = state.metadata.get("targets_per_turn", [])

        parsed_outputs = []

        for turn_output in all_turn_outputs:
            match = answer_pattern.search(turn_output)
            if match:
                content = match.group(1).strip()
                try:
                    parsed_value = int(content)
                    parsed_outputs.append(parsed_value)
                except ValueError:
                    parsed_outputs.append(None)

        turn_results = []
        for turn_output, target in zip(parsed_outputs, targets_per_turn):
            is_correct = (turn_output is not None) and (turn_output == target)
            turn_results.append({"output": turn_output, "target": target, "correct": is_correct})

        turn_accuracy = sum(result["correct"] for result in turn_results) / len(turn_results)

        # Horizon: first turn (0-indexed) where the model was not correct anymore
        # If all turns are correct, horizon is len(turn_results) (number of turns completed)
        horizon = len(turn_results)
        for turn_idx, result in enumerate(turn_results):
            if not result["correct"]:
                horizon = turn_idx
                break

        return Score(
            value={
                "turn_accuracy": turn_accuracy,
                "horizon": horizon,
                "all_correct": all(result["correct"] for result in turn_results),
            },
            answer=str(turn_results),
            explanation=state.output.completion,
        )

    return score


long_horizon_execution_10 = LightevalTaskConfig(
    name="long_horizon_execution",
    prompt_function=lambda line, task_name: line,
    sample_fields=functools.partial(record_to_sample, k=10, max_turns=30),
    solver=[solver()],
    scorer=[scorer()],
    hf_repo="arvindh75/Long-Horizon-Execution",
    hf_subset="default",
    evaluation_splits=("test",),
    metrics=[Metrics.exact_match],
)

TASKS_TABLE = [
    long_horizon_execution_10,
]

"""
name:
Long Horizon Execution

dataset:
arvindh75/Long-Horizon-Execution

abstract:
This dataset is a synthetic benchmark designed to measure the pure execution
capability of LLMs over long horizons. The core task is key-value dictionary addition.
A fixed, in-context dictionary mapping five-letter English words (keys) to integer values
is provided in dictionary.json. The model's goal is to maintain a running sum.
In each turn, it receives one or more keys (defined by the turn complexity, K),
retrieves their corresponding values from the dictionary, adds them to the running sum, and outputs the new sum.
The primary metric for evaluation is the task length: the number of steps a model can execute before its accuracy drops below a certain threshold.

languages:
english

tags:
agent, llm, benchmark

paper:
https://arxiv.org/abs/2509.09677

starred:
true
"""

import re

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def _build_prompt_and_target(record):
    """
    Helper function to extract common logic for building prompt and target.
    Processes the record and returns prompt, target, and metadata.

    Returns:
        tuple: (prompt: str, target_str: str, metadata: dict)
    """
    input_keys = record["input"]
    input_values = record["values"]
    expected_output = record["output"]

    dictionary = dict(zip(input_keys, input_values))
    dict_str = str(dictionary)
    keys_str = str(input_keys)

    prompt = f"""You are an AI assistant. I will provide you with a dictionary and then give you a list of keys.
Your task is to calculate the running cumulative sum (starting from 0) by adding the value associated with each key in order.

For each key in the list, you need to:
1. Look up the value in the dictionary
2. Add it to the running sum
3. Output the cumulative sum after processing all keys up to that point

Dictionary to use:
{dict_str}

Keys to process in order:
{keys_str}

Your task: Calculate the cumulative sum after each key. The first sum is just the value of the first key. The second sum is the first value plus the second value, and so on.

IMPORTANT:
- Output your answer as a single line with comma-separated values inside <answer></answer> tags
- Do not include any other text outside the answer tags
- Format: <answer>value1,value2,value3,...</answer>
- Example: If the cumulative sums are [5, 8, 12], output: <answer>5,8,12</answer>

Your answer:"""

    target_str = ",".join(map(str, expected_output))

    metadata = {
        "input_keys": input_keys,
        "input_values": input_values,
        "expected_output": expected_output,
        "dictionary": dictionary,
        "num_items": len(input_keys),
    }

    return prompt, target_str, metadata


def long_horizon_execution_prompt_function(line, task_name: str = None):
    """
    Prompt function for non-inspect-ai backend.
    Converts dataset record to Doc object.
    """
    prompt, target_str, _ = _build_prompt_and_target(line)

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[target_str],  # Expected answer as a choice
        gold_index=0,
        instruction=prompt,
    )


def record_to_sample(record):
    """
    Converts dataset record to inspect-ai Sample object.
    """
    prompt, target_str, metadata = _build_prompt_and_target(record)

    return Sample(
        input=prompt,
        target=target_str,
        metadata=metadata,
    )


@scorer(metrics={"accuracy": [accuracy(), stderr()]})
def long_horizon_execution_scorer():
    async def score(state: TaskState, target: Target):
        response = state.output.completion

        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        match = answer_pattern.search(response)

        if not match:
            return Score(value="I", answer="", explanation="No <answer> tag found in response.")

        content = match.group(1).strip()

        try:
            pred_values = [int(x.strip()) for x in content.split(",") if x.strip()]
        except ValueError:
            return Score(value="I", answer=content, explanation=f"Failed to parse integers from: {content}")

        try:
            exp_values = [int(x.strip()) for x in target.text.split(",") if x.strip()]

        except (ValueError, AttributeError):
            pred_str = ",".join(map(str, pred_values))
            is_correct = pred_str == target.text
            return Score(
                value="C" if is_correct else "I",
                answer=pred_str,
                explanation=f"Expected: {target.text}, Predicted: {pred_str}",
            )

        is_correct = pred_values == exp_values
        return Score(
            value="C" if is_correct else "I",
            answer=",".join(map(str, pred_values)),
            explanation=(f"Expected {len(exp_values)} values, Got {len(pred_values)} values. Match: {is_correct}"),
        )

    return score


long_horizon_execution = LightevalTaskConfig(
    name="long_horizon_execution",
    prompt_function=long_horizon_execution_prompt_function,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=long_horizon_execution_scorer(),
    hf_repo="arvindh75/Long-Horizon-Execution",
    hf_subset="default",
    evaluation_splits=("test",),
    generation_size=32768,
    metrics=[Metrics.exact_match],
)

TASKS_TABLE = [long_horizon_execution]

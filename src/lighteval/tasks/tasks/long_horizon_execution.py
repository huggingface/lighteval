"""
DOCSTRING TO BE IMPLEMENTED
"""

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig


def record_to_sample(record):
    input_keys = record["input"]
    input_values = record["values"]
    expected_output = record["output"]

    MAX_ITEMS = 100  # for truncation, can be adjusted.
    if len(input_keys) > MAX_ITEMS:
        input_keys = input_keys[:MAX_ITEMS]
        input_values = input_values[:MAX_ITEMS]
        expected_output = expected_output[:MAX_ITEMS]

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

    return Sample(
        input=prompt,
        target=target_str,
        metadata={
            "input_keys": input_keys,
            "input_values": input_values,
            "expected_output": expected_output,
            "dictionary": dictionary,
            "num_items": len(input_keys),
        },
    )


long_horizon_execution = LightevalTaskConfig(
    name="long_horizon_execution",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    hf_repo="arvindh75/Long-Horizon-Execution",
    hf_subset="default",
    evaluation_splits=("test",),
)

TASKS_TABLE = [long_horizon_execution]

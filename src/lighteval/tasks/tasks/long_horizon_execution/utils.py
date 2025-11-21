"""
Utility functions for Long Horizon Execution task.
"""

from lighteval.tasks.tasks.long_horizon_execution.constants import (
    PROMPT_TEMPLATE_MULTI_START,
    PROMPT_TEMPLATE_SINGLE,
)


def _binary_search_max_items(input_keys, build_prompt_fn, prompt_length, min_items=1):
    """
    Generic binary search to find maximum number of items that fit within prompt_length.
    Returns:
        int: Maximum number of items that fit
    """
    # Pre-validate that at least min_items fit within prompt_length
    test_prompt = build_prompt_fn(min_items)
    if test_prompt is None:
        raise ValueError("Cannot build prompt: unable to generate prompt with available items")

    if len(test_prompt) > prompt_length:
        item_label = "item" if min_items == 1 else f"{min_items} items"
        raise ValueError(
            f"Prompt length ({prompt_length} chars) is too small to fit {item_label}. "
            f"Minimum required: {len(test_prompt)} chars. "
            f"Please increase prompt_length or reduce dataset complexity."
        )

    # Binary search to find maximum n that fits within prompt_length
    left, right = min_items, len(input_keys)
    max_n = min_items

    while left <= right:
        mid = (left + right) // 2
        prompt = build_prompt_fn(mid)

        if prompt is None:
            right = mid - 1
            continue

        if len(prompt) <= prompt_length:
            max_n = mid
            left = mid + 1
        else:
            right = mid - 1

    return max_n


def _build_prompt_and_target(record, prompt_length=32768, prompt_template=PROMPT_TEMPLATE_SINGLE):
    """
    Helper function to extract common logic for building prompt and target.
    Uses binary search to find the maximum number of items that fit within prompt_length.
    Processes the record and returns prompt, target, and metadata.
    Args:
        record: Dictionary with 'input', 'values', and 'output' keys
        prompt_length: Maximum character length for the prompt. Defaults to 32768.
        prompt_template: Prompt template to use for formatting. Defaults to PROMPT_TEMPLATE_SINGLE.
    Returns:
        tuple: (prompt: str, target_str: str, metadata: dict)
    """
    input_keys = record["input"]
    input_values = record["values"]
    expected_output = record["output"]

    def build_prompt_for_n(n):
        """Build a prompt with the first n items."""
        if n == 0:
            return None
        keys_n = input_keys[:n]
        values_n = input_values[:n]
        dictionary_n = dict(zip(keys_n, values_n))
        dict_str = str(dictionary_n)
        keys_str = str(keys_n)
        return prompt_template.format(dict_str=dict_str, keys_str=keys_str, num_keys=n)

    # Handle empty input case
    if len(input_keys) == 0:
        raise ValueError("Cannot build prompt: no items available in record")

    max_n = _binary_search_max_items(input_keys, build_prompt_for_n, prompt_length, min_items=1)

    # Use the maximum n that fits
    input_keys = input_keys[:max_n]
    input_values = input_values[:max_n]
    expected_output = expected_output[:max_n]

    dictionary = dict(zip(input_keys, input_values))
    dict_str = str(dictionary)
    keys_str = str(input_keys)
    prompt = prompt_template.format(dict_str=dict_str, keys_str=keys_str, num_keys=len(input_keys))

    target_str = str(expected_output[-1])

    metadata = {
        "input_keys": input_keys,
        "input_values": input_values,
        "expected_output": expected_output,
        "dictionary": dictionary,
        "num_items": len(input_keys),
    }

    return prompt, target_str, metadata


def _find_max_items_for_multi_turn(input_keys, input_values, prompt_length, k):
    """
    Find maximum number of items that fit within prompt_length for multi-turn evaluation.
    Uses binary search to find max items where initial prompt (dict + first K keys) fits.
    Returns:
        int: Maximum number of items that fit
    """

    def build_initial_prompt_for_n(n):
        """Build initial prompt with dictionary and first K keys from n total items."""
        if n == 0:
            return None
        keys_n = input_keys[:n]
        values_n = input_values[:n]
        dictionary_n = dict(zip(keys_n, values_n))
        dict_str = str(dictionary_n)

        # First turn has first K keys
        first_turn_keys = keys_n[:k]
        keys_str = ", ".join(first_turn_keys)

        return PROMPT_TEMPLATE_MULTI_START.format(
            dict_str=dict_str, keys_str=keys_str, k=k, num_keys=len(first_turn_keys)
        )

    return _binary_search_max_items(input_keys, build_initial_prompt_for_n, prompt_length, min_items=k)


def _chunk_and_calculate_expected(input_keys, input_values, k):
    """
    Chunk keys into turns of size K and calculate expected cumulative sums per turn.
    Returns:
        tuple: (turn_chunks: list, value_chunks: list, expected_per_turn: list)
    """
    # Chunk keys into turns of size K
    turn_chunks = []
    value_chunks = []
    for i in range(0, len(input_keys), k):
        turn_chunks.append(input_keys[i : i + k])
        value_chunks.append(input_values[i : i + k])

    # Calculate expected cumulative sums for each turn
    expected_per_turn = []
    cumulative_sum = 0
    for values in value_chunks:
        cumulative_sum += sum(values)
        expected_per_turn.append(cumulative_sum)

    return turn_chunks, value_chunks, expected_per_turn


def _build_multi_turn_prompts(record, prompt_length=32768, k=1):
    """
    Build prompts for multi-turn evaluation.
    Uses binary search to find maximum number of items that fit within prompt_length.
    Chunks keys into turns of size K.
    Args:
        record: Dictionary with 'input', 'values', and 'output' keys
        prompt_length: Maximum character length for the prompt. Defaults to 32768.
        k: Turn complexity (number of keys per turn). Defaults to 1.
    Returns:
        tuple: (initial_prompt: str, turn_chunks: list, expected_per_turn: list, metadata: dict)
    """
    input_keys = record["input"]
    input_values = record["values"]
    expected_output = record["output"]

    # Handle empty input case
    if len(input_keys) == 0:
        raise ValueError("Cannot build prompt: no items available in record")

    # Find maximum number of items that fit
    max_n = _find_max_items_for_multi_turn(input_keys, input_values, prompt_length, k)

    # Use the maximum n that fits
    input_keys = input_keys[:max_n]
    input_values = input_values[:max_n]
    expected_output = expected_output[:max_n]

    turn_chunks, value_chunks, expected_per_turn = _chunk_and_calculate_expected(input_keys, input_values, k)

    dictionary = dict(zip(input_keys, input_values))
    dict_str = str(dictionary)

    first_turn_keys_str = ", ".join(turn_chunks[0])
    initial_prompt = PROMPT_TEMPLATE_MULTI_START.format(
        dict_str=dict_str, keys_str=first_turn_keys_str, k=k, num_keys=len(turn_chunks[0])
    )

    metadata = {
        "turn_chunks": turn_chunks,
        "value_chunks": value_chunks,
        "expected_per_turn": expected_per_turn,
        "dictionary": dictionary,
        "k": k,
        "num_turns": len(turn_chunks),
        "num_items": len(input_keys),
    }

    return initial_prompt, turn_chunks, expected_per_turn, metadata

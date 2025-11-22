"""
Constants file reused within the Long Horizon Execution task.
"""

PROMPT_TEMPLATE_SINGLE = """You are an AI assistant. I will provide you with a dictionary and then give you a list of keys.
Your task is to calculate the final cumulative sum after processing all keys in order.
For each key in the list, you need to:
1. Look up the value in the dictionary
2. Add it to the running sum
3. After processing all keys, output the final cumulative sum
Dictionary to use:
{dict_str}
Keys to process in order:
{keys_str}
Your task: Process all keys in order and calculate the final cumulative sum after processing all {num_keys} keys.
IMPORTANT:
- Output your answer as a single integer value inside <answer></answer> tags
- Do not include any other text outside the answer tags
- Format: <answer>final_sum</answer>
- Example: If the final cumulative sum is 42, output: <answer>42</answer>
Your answer:"""

PROMPT_TEMPLATE_MULTI_START = """You are an AI assistant. I will provide you with a dictionary and then give you keys in groups of {k}.
Your task is to keep a running total (starting from 0) by adding the values associated with the keys I provide.
In each turn, I'll provide {k} keys (comma-separated).
Respond with the current running sum, enclosed in <answer> tags.
Dictionary to maintain:
{dict_str}
Ready to start!
**User**: {keys_str}
**Assistant**:"""

PROMPT_TEMPLATE_MULTI_FOLLOWUP = """Here are the next keys to process:
**User**: {keys_str}
**Assistant**:"""

CONTEXT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
TURN_COMPLEXITIES = [1, 2, 10]

# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


ruler_model = os.environ.get("RULER_MODEL")
ruler_org = os.environ.get("RULER_ORG")
if ruler_model is None:
    raise ValueError("RULER_MODEL environment variable is not set, set it to the model you want to evaluate")
if ruler_org is None:
    raise ValueError("RULER_ORG environment variable is not set, set it to the organization you want to evaluate")

subsets = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

lengths = [262144, 131072, 65536, 32768, 16384, 8192, 4096]

tokens_to_generate = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
    "niah_multivalue": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}

patterns = {
    "niah_multikey_1": r"The special magic number for ([\w\s-]+?) mentioned in the provided text is$",
    "niah_multikey_2": r"The special magic number for ([\w\s-]+?) mentioned in the provided text is$",
    "niah_single_1": r"The special magic number for ([\w\s-]+?) mentioned in the provided text is$",
    "niah_single_2": r"The special magic number for ([\w\s-]+?) mentioned in the provided text is$",
    "niah_single_3": r"The special magic uuid for ([\w\s-]+?) mentioned in the provided text is$",
    "niah_multivalue": r"The special magic numbers for ([\w\s-]+?) mentioned in the provided text are$",
    "niah_multiquery": r"The special magic numbers for ((?:[\w\s-]+?, )*(?:and )?[\w\s-]+?) mentioned in the provided text are$",
    "niah_multikey_3": r"The special magic uuid for ([a-fA-F0-9-]{36}) mentioned in the provided text is$",
}


def extract_answer(text, pattern):
    """
    Extract everything after 'Answer:' and return both the cleaned text and the answer.

    Args:
        text (str): The original text containing 'Answer:'

    Returns:
        tuple: (cleaned_text, answer) where answer is None if 'Answer:' not found
    """
    # Regex pattern to match 'Answer:' and capture everything after it
    match = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
    if len(match) > 0:
        match = match[-1]
    else:
        return text, None
    if match:
        # Extract the answer (everything after 'Answer:', answer included)
        answer = match.group(0).strip()
        # Remove 'Answer:' and everything after it from the original text
        cleaned_text = text[: match.start()] + text[match.end() :]
        return cleaned_text, answer
    else:
        # No 'Answer:' found, return original text and None
        return text, None


def ruler(line, task_name: str = ""):
    query = line["input"]
    choices = line["outputs"]
    gold_index = 0
    answer_prefix = line["answer_prefix"]
    return Doc(
        query=query.strip(),
        choices=choices,
        gold_index=gold_index,
        task_name=task_name,
        specific={"answer_prefix": answer_prefix},
    )


task_configs = []

for subset in subsets:
    for length in lengths:
        task_configs.append(
            LightevalTaskConfig(
                name=f"ruler_{length}:{subset}",
                suite=["lighteval"],
                prompt_function=ruler,
                hf_repo=f"{ruler_org}/RULER-{length}-{ruler_model}",
                hf_subset="default",
                hf_avail_splits=[subset],
                evaluation_splits=[subset],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=tokens_to_generate[subset],
                metric=[Metrics.ruler_match_any] if subset in ["qa_1", "qa_2"] else [Metrics.ruler_match_all],
                stop_sequence=None,
                trust_dataset=False,
                version=0,
            )
        )

TASKS_TABLE = task_configs

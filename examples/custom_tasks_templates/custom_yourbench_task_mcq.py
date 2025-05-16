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


import logging
import re

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


def extract_answer_letter_from_pred(line: str) -> str | None:
    # Find all standalone A–D letters, and use the last one
    matches = re.findall(r"\b([A-Da-d])\b", line)
    if matches:
        pred = matches[-1].upper()
        return pred
    return None


def extract_answer_letter_from_gold(line: str) -> str | None:
    # Match (A), (B) OR lines starting with A., B.
    match = re.match(r"^\(?([A-Da-d0-9])\)?[).]?\s+", line.strip())
    if match:
        gold = match.group(1).upper()
        return gold
    return None


ZEROSHOT_QA_USER_PROMPT = """
Answer the following multiple-choice question by selecting only one letter: A, B, C, or D. Do not explain your answer.

Question: {question}

Choices:
{options}

Answer:
"""


def yourbench_prompt(line, task_name: str = ""):
    options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(line["choices"]))

    gold_raw = line["gold"][0]

    if isinstance(gold_raw, str) and gold_raw.strip().isalpha():
        gold_index = ord(gold_raw.strip().upper()) - ord("A")
    elif isinstance(gold_raw, int):
        gold_index = gold_raw
    else:
        raise ValueError(f"Unexpected gold label format: {gold_raw!r}")

    return Doc(
        task_name=task_name,
        query=ZEROSHOT_QA_USER_PROMPT.format(question=line["question"], options=options),
        choices=line["choices"],
        gold_index=gold_index,
    )


yourbench_metrics = SampleLevelMetric(
    metric_name="mcq_accuracy",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=ExactMatches(
        normalize_gold=extract_answer_letter_from_gold, normalize_pred=extract_answer_letter_from_pred
    ).compute,
    corpus_level_fn=np.mean,
)

extend_enum(Metrics, "yourbench_metrics", yourbench_metrics)

yourbench_mcq = LightevalTaskConfig(
    name="HF_TASK_NAME",  # noqa: F821
    suite=["custom"],
    prompt_function=yourbench_prompt,
    hf_repo="HF_DATASET_NAME",  # noqa: F821
    hf_subset="lighteval",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metric=[Metrics.yourbench_metrics],
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = [yourbench_mcq]

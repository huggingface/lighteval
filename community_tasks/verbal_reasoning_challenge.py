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

# ruff: noqa: F405, F403, F401
"""
Task to evaluate LLMs on the verbal reasoning challenge dataset:
https://huggingface.co/datasets/nuprl/verbal-reasoning-challenge

"""

import re
from typing import List

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def verbal_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["challenge"],
        choices=[line["answer"]],
        gold_index=0,
        specific={"ID": line["ID"]},
    )


def _parse_answer(text: str) -> List[List[str]]:
    """
    Converts text to lowercase. Then interprets ";" as a separator between
    alternatives. Within each alternative, interprets "," and "-->" as separators
    for elements of a set. Within each set, drops all non-alphanumeric characters
    and returns that set.

    Another way to describe this is that we interpret adjacent words as
    phrases that must be present literally. However, comma and arrow separate
    distinct phrases that may be present in any order. All other characters
    are dropped.
    """
    text = text.lower()
    alternatives = re.split(r";", text)
    result = []
    for alternative in alternatives:
        groups = re.split(r"–?-?-?>|,", alternative)
        result.append([" ".join(re.findall(r"\b\w+\b", group)) for group in groups])
    return result


def _answer_without_thoughts(completion: str) -> str:
    completion = re.sub(r"(<think>)?[^<]*<\/think>", "", completion).strip()
    return completion


def _check_answer(completion: str, answer: str) -> bool:
    """
    Check that all the phrases that must appear in the answer appear in the
    completion. We ignore "thoughts", capitalization, and punctuation.
    """
    completion = _answer_without_thoughts(completion).lower()
    completion = re.sub(r"[^\w\s]", " ", completion)
    completion = re.sub(r"\s+", " ", completion)
    alternative_answers = _parse_answer(answer)
    for answer_phrases in alternative_answers:
        if all(re.search(rf"\b{re.escape(phrase)}\b", completion) for phrase in answer_phrases):
            return True
    return False


def verbal_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    completion = predictions[0]
    answer = formatted_doc.choices[formatted_doc.gold_index]
    return _check_answer(completion, answer)


verbal_custom_metric = SampleLevelMetric(
    metric_name="Verbal_Metric",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=verbal_metric,
    corpus_level_fn=np.mean,
)


task = LightevalTaskConfig(
    name="verbal_reasoning_challenge",
    prompt_function=verbal_prompt_fn,
    suite=["community"],
    hf_repo="nuprl/verbal-reasoning-challenge",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metric=[verbal_custom_metric],
)

TASKS_TABLE = [task]

extend_enum(Metrics, "verbal_custom_metric", verbal_custom_metric)

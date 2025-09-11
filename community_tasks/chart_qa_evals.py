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
Task to evaluate VLMs on HuggingFaceM4/ChartQA.

Example evaluation:
lighteval accelerate "model_name=google/gemma-3-4b-it" "community|chart_qa|0" --custom-tasks community_tasks/chart_qa_evals.py --vision-model
"""

import numpy as np

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)
from lighteval.metrics.utils.metric_utils import SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def prompt_fn(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """
    return Doc(
        task_name=task_name,
        query="Answer the following question. The last line of your response should be of the following format: 'Answer: $ANSWER' (without quotes) where $ANSWER is the answer to the question.\n\n"
        + line["query"],
        gold_index=0,
        choices=[line["label"]],
        images=[line["image"]],
    )


extraction_targets = [ExprExtractionConfig(), LatexExtractionConfig()]
metric = SampleLevelMetric(
    metric_name="extractive_match",
    sample_level_fn=MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        gold_extraction_target=extraction_targets,
        pred_extraction_target=extraction_targets,
        precision=6,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

task = LightevalTaskConfig(
    name="chart_qa",
    prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="HuggingFaceM4/ChartQA",
    hf_subset="default",
    hf_avail_splits=["train", "val", "test"],
    evaluation_splits=["test"],
    hf_filter=lambda line: line["human_or_machine"] == 0,
    few_shots_split=None,
    few_shots_select=None,
    metrics=[metric],  # select your metric in Metrics
)

human_task = LightevalTaskConfig(
    name="chart_qa:human",
    prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="HuggingFaceM4/ChartQA",
    hf_subset="default",
    hf_avail_splits=["train", "val", "test"],
    evaluation_splits=["test"],
    hf_filter=lambda line: line["human_or_machine"] == 0,
    few_shots_split=None,
    few_shots_select=None,
    metrics=[metric],  # select your metric in Metrics
)

TASKS_TABLE = [task, human_task]

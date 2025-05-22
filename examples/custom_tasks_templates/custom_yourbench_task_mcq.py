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

from aenum import extend_enum

from lighteval.metrics.dynamic_metrics import multilingual_extractive_match_metric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


logger = logging.getLogger(__name__)


ZEROSHOT_QA_INSTRUCTION = """
Answer the following multiple-choice question by selecting only one letter: A, B, C, or D. Do not explain your answer.
"""

ZEROSHOT_QA_USER_PROMPT = (
    ZEROSHOT_QA_INSTRUCTION
    + """
Question: {question}

Choices:
{options}

Answer:
"""
)


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
        instruction=ZEROSHOT_QA_INSTRUCTION,
        task_name=task_name,
        query=ZEROSHOT_QA_USER_PROMPT.format(question=line["question"], options=options),
        choices=line["choices"],
        gold_index=gold_index,
    )


yourbench_metrics = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=6,
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

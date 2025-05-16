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
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetricGrouping,
    MetricCategory,
    MetricUseCase,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


def extract_content_from_xml_tags(full_content, xml_tag):
    # This function extracts the content between the XML tags
    # It uses regex to find the content and includes error handling

    # Define the regex patterns to match the content
    pattern_with_closing_tag = f"<{xml_tag}>(.*?)</{xml_tag}>"
    pattern_without_closing_tag = f"<{xml_tag}>(.*)"

    try:
        # First, try to find matches with both opening and closing tags
        matches_with_closing = re.findall(pattern_with_closing_tag, full_content, re.DOTALL)
        if matches_with_closing:
            return matches_with_closing[0].strip()

        # If no matches found, try to find content with only opening tag
        matches_without_closing = re.findall(pattern_without_closing_tag, full_content, re.DOTALL)
        if matches_without_closing:
            return matches_without_closing[0].strip()

        # If still no matches found, return an empty string
        return ""

    except Exception as extraction_error:
        logger.error(f"Error extracting content from XML tags: {extraction_error}")
        return ""


def custom_metric_compute(golds, predictions, formatted_doc) -> list[dict[str, float]]:
    try:
        answer = extract_content_from_xml_tags(predictions[0], xml_tag="answer")
        if answer and formatted_doc.gold_index == ord(str(answer)) - ord("A"):
            return {"accuracy": 1}
    except Exception as e:
        logger.error(f"Error during parsing the response {type(e)} {e}")

    return {"accuracy": 0}


# Prompt template for zero-shot MCQ
ZEROSHOT_QA_USER_PROMPT = """You are given a question multiple-choice question. Select one correct answer from the provided options. Respond with the letter of the correct choice inside <answer> tags.

<question>
{question}
</question>

<options>
{options}
</options>

Enclose your answer like this:
<answer>
B
</answer>"""


def yourbench_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=ZEROSHOT_QA_USER_PROMPT.format(question=line["question"], options="\n".join(line["choices"])),
        choices=line["choices"],
        gold_index=line["gold"][0],
        specific={
            "question_category": line["question_category"],
            "kind": line["kind"],
            "estimated_difficulty": line["estimated_difficulty"],
            "document_id": line["document_id"],
            "question_generating_model": line["question_generating_model"],
            "chunks": line["chunks"],
            "question": line["question"],
            "document": line["document"],
        },
    )


yourbench_metrics = CorpusLevelMetricGrouping(
    metric_name=["accuracy"],
    higher_is_better={"accuracy": True},
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=custom_metric_compute,
    corpus_level_fn={"accuracy": np.mean},
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
    stop_sequence=["</answer>"],
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = [yourbench_mcq]

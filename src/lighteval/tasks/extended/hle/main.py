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
import math
from typing import List, Literal

import numpy as np
from aenum import extend_enum
from pydantic import BaseModel

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetricGrouping,
    MetricCategory,
    MetricUseCase,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]  # 100% reliability


"""Adaptation from https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py
"""


def get_judge_prompt(question: str, answer: str, gold: str, **kwargs):
    return [
        {
            "role": "user",
            "content": rf"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {gold}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.""",
        },
    ]


def process_judge_response_hle(response: ExtractedAnswer | List[ExtractedAnswer]):
    # todo: add support for batched responses
    if isinstance(response, list):
        response = response[0]
    return {
        # "correct_answer": correct_answer,
        "model_answer": response.extracted_final_answer,
        "reasoning": response.reasoning,
        "correct": response.correct,
        "confidence": response.confidence,
    }


class JudgeLLMHLE(JudgeLLM):
    def __init__(self):
        super().__init__(
            judge_model_name="gpt-4o-2024-08-06",
            template=get_judge_prompt,
            process_judge_response=process_judge_response_hle,
            judge_backend="openai",
            short_judge_name="hle_judge",
            response_format=ExtractedAnswer,
        )

    def compute(self, sample_ids: list[str], responses: list, formatted_docs: list[Doc]) -> list[dict[str, float]]:
        # If we are evaluating a multiturn task, we need to have specific field in the formatted doc
        questions = [formatted_doc.specific["question"] for formatted_doc in formatted_docs]
        golds = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]
        predictions = [response[0].result[0] for response in responses]
        options = [None] * len(questions)

        score, _, _ = self.judge.evaluate_answer_batch(questions, predictions, options, golds)

        metrics = []
        for i in range(len(sample_ids)):
            score[i]["correct_answer"] = golds[i]
            metrics.append(
                {
                    "accuracy": score[i],
                    "confidence_half_width": score[i],
                    "calibration_error": score[i],
                }
            )

        return metrics

    def compute_corpus(self, scores: List[dict]):
        n = len(scores)

        correct = []
        confidence = []
        for score in scores:
            correct.append("yes" in score["correct"])
            confidence.append(score["confidence"])

        correct = np.array(correct)
        confidence = np.array(confidence)

        # sometimes model collapses on same questions
        if len(correct) != n:
            print(f"Available predictions: {len(correct)} | Total questions: {n}")

        accuracy = round(100 * sum(correct) / n, 2)
        # Wald estimator, 95% confidence interval
        confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)
        calibration_error = round(calib_err(confidence, correct, p="2", beta=100), 2)

        return {
            "accuracy": accuracy,
            "confidence_half_width": confidence_half_width,
            "calibration_error": calibration_error,
        }


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p="2", beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if len(bins) == 0:
        logger.warning("Error when computing the bins for calibration error")
        return -1

    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p == "infty" or p == "infinity" or p == "max":
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == "2":
        cerr = np.sqrt(cerr)

    return cerr


def hle_text_only(line, task_name: str = None):
    if line["image"] not in [None, ""]:
        return

    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[line["answer"]],
        gold_index=0,
        specific={"question": line["question"]},
    )


hle_metrics = CorpusLevelMetricGrouping(
    metric_name=["accuracy", "confidence_half_width", "calibration_error"],
    higher_is_better=dict.fromkeys(["accuracy", "confidence_half_width", "calibration_error"], True),
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=JudgeLLMHLE().compute,
    corpus_level_fn=JudgeLLMHLE().compute_corpus,
)
extend_enum(Metrics, "hle_metrics", hle_metrics)

hle = LightevalTaskConfig(
    name="hle",
    suite=["lighteval"],
    prompt_function=hle_text_only,
    hf_repo="cais/hle",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metric=[Metrics.exact_match, Metrics.hle_metrics],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)


TASKS_TABLE = [hle]

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

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    MetricCategory,
    MetricUseCase,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)

JUDGE_ANSWER_SYSTEM_PROMPT = """You will be provided with the summary of a document, a piece of text, a question generated from that text, and the correct or "gold" answer to the question. Additionally, you will receive two answers: Answer A and Answer B. Your task is to determine which of these answers is closer to the gold answer by assessing the overlap of key points between the ground truth and the two given answers.

# Steps

1. **Document Understanding**:
   - Analyze the provided document summary to grasp the context and main themes.

2. **Chunk Understanding**:
   - Examine the provided text (chunk) to understand its content.

3. **Question Understanding**:
   - Interpret the given question to fully comprehend what is being asked.

4. **Ground Truth Answer Understanding**:
   - Understand the provided ground truth answer, identifying its key points.

5. **Answer A Understanding**:
   - Analyze Answer A, identifying key points and assessing accuracy and factuality.

6. **Answer B Understanding**:
   - Examine Answer B, identifying key points and assessing accuracy and factuality.

7. **Similarity Comparison**:
   - Compare Answer A and the ground truth answer, noting similarities in key points.
   - Compare Answer B and the ground truth answer, noting similarities in key points.

8. **Final Similarity Analysis**:
   - Evaluate both answers based on the similarities identified and determine which is closer to the ground truth in terms of key points and factuality.

# Output Format

- Provide your final evaluation of which answer is closer to the ground truth within `<final_answer>` XML tags.
- Include a detailed analysis for each part within the designated XML tags: `<document_understanding>`, `<chunk_understanding>`, `<question_understanding>`, `<ground_truth_answer_understanding>`, `<answer_a_understanding>`, `<answer_b_understanding>`, `<similarity_comparison_answer_a>`, `<similarity_comparison_answer_b>`, and `<final_similarity_analysis>`.

# Examples

**Input**:
```xml
<document_summary>
[Summary]
</document_summary>

<piece_of_text>
[Text]
</piece_of_text>

<question>
[Question]
</question>

<gold_answer>
[Gold Answer]
</gold_answer>

<answer_a>
[Answer A]
</answer_a>

<answer_b>
[Answer B]
</answer_b>
```
**Output**:
```xml

<document_understanding>
Understanding of the summary including key themes
</document_understanding>

<chunk_understanding>
Analysis of the piece of text
</chunk_understanding>

<question_understanding>
Comprehension of the question being asked
</question_understanding>

<ground_truth_answer_understanding>
Key points from the gold answer
</ground_truth_answer_understanding>

<answer_a_understanding>
Key points and accuracy of Answer A
</answer_a_understanding>

<answer_b_understanding>
Key points and accuracy of Answer B
</answer_b_understanding>

<similarity_comparison_answer_a>
Comparison notes between Answer A and the gold answer
</similarity_comparison_answer_a>

<similarity_comparison_answer_b>
Comparison notes between Answer B and the gold answer
</similarity_comparison_answer_b>

<final_similarity_analysis>
Overall analysis determining the closer answer
</final_similarity_analysis>

<final_answer>
Answer X (where X is the option you pick)
</final_answer>
```

# Notes

- Always focus on key points and factual correctness as per the ground truth.
- Avoid any biases and rely solely on the evidence presented.
- Enclose all evaluations and analyses in the specified XML tags for clarity and structure."""


JUDGE_ANSWER_USER_PROMPT = """<document_summary>
{summary}
</document_summary>

<piece_of_text>
{chunk}
</piece_of_text>

<question>
{question}
</question>

<gold_answer>
{oracle_answer}
</gold_answer>

<answer_a>
{answer_a}
</answer_a>

<answer_b>
{answer_b}
</answer_b>"""


def get_judge_prompt(question: str, answer: str, gold: str, **kwargs):
    chunk = kwargs.get("chunk", "")
    summary = kwargs.get("summary", "")

    return [
        {"role": "system", "content": JUDGE_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_ANSWER_USER_PROMPT.format(
                summary=summary, chunk=chunk, question=question, oracle_answer=gold, answer_a=answer, answer_b=answer
            ),
        },
    ]


def process_judge_response_yourbench(response):
    return 1


class JudgeLLMYourBench(JudgeLLM):
    def __init__(self):
        super().__init__(
            judge_model_name="gpt-4o-2024-08-06",
            template=get_judge_prompt,
            process_judge_response=process_judge_response_yourbench,
            judge_backend="openai",
            short_judge_name="yourbench_judge",
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


ZEROSHOT_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""


def yourbench_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=ZEROSHOT_QA_USER_PROMPT.format(question=line["question"]),
        choices=["EXPECTED ANSWER"],
        gold_index=0,
        specific={
            "chunk": line["chunk"],
            "summary": line["summary"],
            "question": line["question"],
            "oracle_answer": line["answer"],
        },
    )


yourbench_metrics = CorpusLevelMetric(
    metric_name="accuracy",
    higher_is_better=True,
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=JudgeLLMYourBench().compute,
    corpus_level_fn=np.mean,
)
extend_enum(Metrics, "yourbench_metrics", yourbench_metrics)

yourbench = LightevalTaskConfig(
    name="hle",
    suite=["lighteval"],
    prompt_function=yourbench_prompt,
    hf_repo=HF_DATASET_NAME,  # noqa: F821
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


TASKS_TABLE = [yourbench]

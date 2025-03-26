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
from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetricGrouping,
    MetricCategory,
    MetricUseCase,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)

JUDGE_ANSWER_SYSTEM_PROMPT = """You will be provided with the summary of a document, a piece of text, a question generated from that text, and the correct or "gold" answer to the question. Additionally, you will receive a model answer. Your task is to determine wether the model answer is correct using the provided "gold" answer as a reference.

# Steps

1. **Document Understanding**:
   - Analyze the provided document summary to grasp the context and main themes.

2. **Chunk Understanding**:
   - Examine the provided text (chunk) to understand its content.

3. **Question Understanding**:
   - Interpret the given question to fully comprehend what is being asked.

4. **Ground Truth Answer Understanding**:
   - Understand the provided ground truth answer, identifying its key points.

5. **Answer Understanding**:
   - Examine the Model Answer, identifying key points and assessing accuracy and factuality.

6. **Final Answer**:
   - 0 or 1 (0 if the model answer is incorrect, 1 if it is correct).

# Output Format

- Provide your final evaluation of whether the answer is correct within `<final_answer>` XML tags.
- Include a detailed analysis for each part within the designated XML tags: `<document_understanding>`, `<chunk_understanding>`, `<question_understanding>`, `<ground_truth_answer_understanding>`, `<model_answer_understanding>`, and `<final_answer>`.

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

<model_answer>
[Model Answer]
</model_answer>
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

<model_answer_understanding>
Key points and accuracy of Answer A
</model_answer_understanding>

<final_answer>
1 or 0 (1 if the model answer is correct, 0 if it is incorrect)
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

<model_answer>
{model_answer}
</model_answer>"""


def get_judge_prompt(question: str, answer: str, gold: str, **kwargs):
    chunk = kwargs.get("chunks", "")
    summary = kwargs.get("documents", "")

    return [
        {"role": "system", "content": JUDGE_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_ANSWER_USER_PROMPT.format(
                summary=summary, chunk=chunk, question=question, oracle_answer=gold, model_answer=answer
            ),
        },
    ]


def process_judge_response_yourbench(response):
    # extract the final answer using regex from the response xml
    try:
        answer = re.search(r"<final_answer>(.*?)</final_answer>", response, re.DOTALL).group(1)
        return int(answer)
    except Exception as e:
        logger.error(f"Error processing judge response: {e}")
    return 0


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
        chunks = [formatted_doc.specific["chunks"][0] for formatted_doc in formatted_docs]
        documents = [formatted_doc.specific["document"] for formatted_doc in formatted_docs]

        score, _, _ = self.judge.evaluate_answer_batch(
            questions, predictions, options, golds, chunks=chunks, documents=documents
        )

        metrics = []
        for i in range(len(sample_ids)):
            metrics.append(
                {
                    "accuracy": score[i],
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
        choices=[line["ground_truth_answer"]],
        gold_index=0,
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
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=JudgeLLMYourBench().compute,
    corpus_level_fn={"accuracy": np.mean},
)
extend_enum(Metrics, "yourbench_metrics", yourbench_metrics)

yourbench = LightevalTaskConfig(
    name=HF_TASK_NAME,  # noqa: F821
    suite=["custom"],
    prompt_function=yourbench_prompt,
    hf_repo=HF_DATASET_NAME,  # noqa: F821
    hf_subset="lighteval_single_shot_questions",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metric=[Metrics.yourbench_metrics],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)


TASKS_TABLE = [yourbench]

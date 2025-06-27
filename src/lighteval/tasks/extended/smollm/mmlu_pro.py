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


import numpy as np

from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.metrics.metrics_sample import (
    PassAtK,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def mmlu_pro(line, task_name: str = None):
    num_choices = len(line["options"])
    # GPQA style
    instruction = f"Given the following question about {line['category']} and answer choices, output the letter corresponding to the correct answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {', '.join(LETTER_INDICES[: num_choices - 1])}, or {LETTER_INDICES[num_choices]}. Think step by step before answering.\n\n"
    query = f"{instruction}###\nQuery:\n{line['question']}\n###\nChoices:"
    query += "".join([f"\n{key}) {choice}" for key, choice in zip(LETTER_INDICES, line["options"])])
    query += "\n###\n"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[:num_choices],
        gold_index=line["answer_index"],
        instruction=instruction,
    )


mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["extended"],
    prompt_function=mmlu_pro,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select=None,
    generation_size=30000,  # needed for reasoning models like R1
    stop_sequence=[],  # no stop sequence, will use eos token
    metric=[
        SampleLevelMetric(
            metric_name="pass@1:1_samples",
            sample_level_fn=PassAtK(
                k=1,
                n=1,
                sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
                    language=Language.ENGLISH,
                    gold_extraction_target=[
                        IndicesExtractionConfig(
                            prefix_for_extraction="NativeLetters", try_extract_without_anchor=False
                        )
                    ],
                    pred_extraction_target=[
                        IndicesExtractionConfig(
                            prefix_for_extraction="NativeLetters", try_extract_without_anchor=False
                        )
                    ],
                    precision=6,
                ).sample_level_fn([ref], [pred], doc),
            ).compute,
            category=MetricCategory.GENERATIVE_SAMPLING,
            use_case=MetricUseCase.REASONING,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )
    ],
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = [mmlu_pro]

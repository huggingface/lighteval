# MIT License

# Copyright (c) 2025 The HuggingFace Team

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
"""Usage:

lighteval vllm pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,tensor_parallel_size=1,max_model_length=32768,gpu_memory_utilisation=0.8 "extended|gpqa:diamond|0|0" \
    --use-chat-template \
    --custom-tasks src/lighteval/tasks/extended/gpqa/main.py
"""
import random

from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig
from lighteval.utils.language import Language


def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    instruction = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
    query = f"{instruction}\n\n{line['Question']}\n\n" ""
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


extraction_targets = [IndicesExtractionConfig(prefix_for_extraction="NativeLetters")]

metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=extraction_targets,
    pred_extraction_target=extraction_targets,
    precision=6,
)

subsets = ["extended", "main", "diamond"]
task_configs = []

for subset in subsets:
    task = LightevalTaskConfig(
        name=f"gpqa:{subset}",
        suite=["extended"],
        prompt_function=gpqa_prompt_fn,
        hf_repo="Idavidrein/gpqa",
        hf_subset="gpqa_diamond",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select="random_sampling",
        generation_size=32_000,
        metric=[metric],
        stop_sequence=[],  # no stop sequence, will use eos token
        trust_dataset=True,
        version=0,
    )
    task_configs.append(task)

TASKS_TABLE = task_configs

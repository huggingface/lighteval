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
"""
Code obtained from: https://github.com/QwenLM/CodeElo

Usage:
lighteval vllm \
    "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=bfloat16,data_parallel_size=8,max_model_length=32768,gpu_memory_utilisation=0.8,generation_parameters={temperature: 0.7,repetition_penalty:1.1,top_p:0.8,top_k:20}" \
    "extended|codeelo:rating|0|0" \
    --use-chat-template
"""

from typing import Any

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import MetricCategory, Metrics, MetricUseCase, SampleLevelMetric
from lighteval.tasks.extended.codeelo.utils import LANG_MAP, extract_code_blocks, make_html_problem, submit_code
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


def codeelo_prompt_fn(line: dict[str, Any], task_name: str = "codeelo:rating") -> Doc:
    instruction = "You are a coding expert. Given a competition-level coding problem, you need to write a C++ program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```."
    return Doc(
        task_name=task_name,
        query=f"{instruction}\n\n{make_html_problem(line)}",
        choices=[""],
        gold_index=0,
        specific={
            "problem_id": line["id"],
        },
    )


def codeelo(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    """Estimates the Pass@1 metric for the code generation task.
    Extract the code from each prediction, Runs it for each sample and generations,
    and computes the Pass@1 over the outputs.
    """
    code_blocks = [extract_code_blocks(pred)[0] for pred in predictions]
    # Extract the code and language
    # TODO: For the moment we are only considering a single generation per problem, there will be 8 (like Pass@1:8)
    for code_block in code_blocks:
        if not code_block:
            return 0.0
        code, lang = code_block
        lang_id = LANG_MAP.get(lang, None)
        if not lang_id:
            return 0.0
        submission_id = submit_code(formatted_doc.specific["problem_id"], lang_id, code, tag="")
        if isinstance(submission_id, str):
            return 0.0

    return 0.0


codeelo_metric = SampleLevelMetric(
    metric_name="codeelo_rating@8",  # Generate 8 outputs per problem
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=codeelo,
    corpus_level_fn=np.mean,
)


extend_enum(Metrics, "codeelo_metric", codeelo_metric)


task = LightevalTaskConfig(
    name="codeelo:rating",
    suite=["extended"],
    prompt_function=codeelo_prompt_fn,
    hf_repo="Qwen/CodeElo",
    hf_subset="test",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    generation_size=32768,
    metric=[Metrics.codeelo_metric],
    stop_sequence=[],  # no stop sequence, will use EOS token
    version=0,
)


TASKS_TABLE = [task]

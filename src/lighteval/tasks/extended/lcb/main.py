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
lighteval vllm \
    "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,tensor_parallel_size=4,max_model_length=32768,gpu_memory_utilisation=0.8" \
    "extended|lcb:codegeneration|0|0" \
    --custom-tasks src/lighteval/tasks/extended/lcb/main.py
"""

import json
from typing import Any

import numpy as np
from aenum import extend_enum

import lighteval.tasks.extended.lcb.lcb_utils as lcb_utils
from lighteval.metrics.metrics import MetricCategory, Metrics, MetricUseCase, PassAtK, SampleLevelMetric
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


SYSTEM_MESSAGE_DEEPSEEK_R1 = "<｜begin▁of▁sentence｜>A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"


def prepare_prompt(line: dict[str, Any]) -> str:
    query = SYSTEM_MESSAGE_DEEPSEEK_R1 + "\n\n"
    query += "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    query += "<｜Assistant｜>"
    return query


def lcb_codegeneration_prompt_fn(line, task_name: str = "lcb:codegeneration") -> Doc:
    """ """
    # For the prompt we need a more general function that can be used tweaked like in:
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    query = prepare_prompt(line)
    # List of dicts of the form: [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = lcb_utils.translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        instruction="",
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": line["metadata"].get("func_name", None),
            # To determine how to run the function
            "is_stdin": any(test["testtype"] == "stdin" for test in inputs),
            # "contest_date": line["contest_date"].isoformat(),  # NOTE: This should be used to filter the dataset
        },
    )


def lcb_codegen_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> SampleLevelMetric:
    """LiveCodeBench code generation metric.
    Steps:
    1. Extract the code from the prediction
    2. Run the code on the inputs
    3. Compute the Pass@1 over the outputs
    """
    # Extract generated code snippets
    generated_code_snippets = [lcb_utils.extract_code(pred) for pred in predictions]  # noqa: F841
    evaluation_samples = {  # noqa: F841
        "inputs": formatted_doc.specific["inputs"],
        "outputs": formatted_doc.specific["outputs"],
        "fn_name": formatted_doc.specific["fn_name"],
    }

    def codegen_metrics(reference: str, generated: str) -> dict[str, Any]:
        return {"passes": 1}

    n = len(predictions)  # The LiveCodeBench repo uses 10 by default

    return SampleLevelMetric(
        metric_name=f"pass@1:{n}_samples",
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.REASONING,
        higher_is_better=True,
        sample_level_fn=PassAtK(
            k=kwargs.get("k", 1), n=len(predictions), sample_scoring_function=codegen_metrics
        ).compute,
        corpus_level_fn=np.mean,
    )


extend_enum(Metrics, "lcb_codegen_metric", lcb_codegen_metric)


task = LightevalTaskConfig(
    name="lcb:codegeneration",
    suite=["extended"],
    prompt_function=lcb_codegeneration_prompt_fn,
    # Needs the version_tag argument, and an additional filter to avoid running on all the examples
    hf_repo="livecodebench/code_generation_lite",
    hf_subset="default",
    hf_avail_splits=["test"],
    # TODO: We need a way of filtering data passing tuple start/end date to have a more fine-grained control of the subset
    # evaluated
    hf_version_tag="v3_v5",  # v3_v5 is the version with the new test cases corresponding to the R1 models.
    evaluation_splits=["test"],
    generation_size=32768,
    metric=[Metrics.lcb_codegen_metric],
    stop_sequence=[],  # no stop sequence, will use EOS token
    trust_dataset=True,
    version=0,
)


TASKS_TABLE = [task]

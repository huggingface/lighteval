"""
name:
Live Code Bench

dataset:
lighteval/code_generation_lite

abstract:
LiveCodeBench collects problems from periodic contests on LeetCode, AtCoder, and
Codeforces platforms and uses them for constructing a holistic benchmark for
evaluating Code LLMs across variety of code-related scenarios continuously over
time.

languages:
english

tags:
code-generation

paper:
https://livecodebench.github.io/
"""

import json
from textwrap import dedent
from typing import Any

import numpy as np
from aenum import extend_enum
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks.lcb.codegen_metrics import (
    codegen_metrics,
    extract_code,
    run_test,
    translate_private_test_cases,
)


def prepare_prompt(line: dict[str, Any]) -> str:
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    return query


def lcb_codegeneration_prompt_fn(line, task_name: str = "lcb:codegeneration") -> Doc:
    # For the prompt we need a more general function that can be used tweaked like in:
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    query = prepare_prompt(line)
    # List of dicts of the form: [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(line["metadata"]).get("func_name", None),
        },
    )


class CodegenMetric(SampleLevelComputation):
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        """Estimates the Pass@1 metric for the code generation task.
        Extract the code from each prediction, Runs it for each sample and generations,
        and computes the Pass@1 over the outputs.
        """
        assert doc.specific is not None, "Doc specific field is required for codegen_metric"

        predictions = model_response.final_text
        # Extract generated code snippets
        generated_code_snippets = [[extract_code(pred) for pred in predictions]]  # noqa: F841
        evaluation_sample = {  # noqa: F841
            "inputs": doc.specific["inputs"],
            "outputs": doc.specific["outputs"],
            "fn_name": doc.specific["fn_name"],
        }
        # This is a list of lists because
        evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

        metrics, _ = codegen_metrics(
            evaluation_sample,
            generated_code_snippets,
            k_list=[1],  # Only run for Pass@1
            num_process_evaluate=8,
        )
        return metrics["pass@1"]


lcb_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",  # This is the way of informing the number of generations currently
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=CodegenMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


extend_enum(Metrics, "lcb_codegen_metric", lcb_codegen_metric)

configs = [
    "release_v1",
    "release_v2",
    "release_v3",
    "release_v4",
    "release_v5",
    "release_v6",
    "release_latest",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v1_v2",
    "v1_v3",
    "v1_v4",
    "v1_v5",
    "v2_v3",
    "v2_v4",
    "v2_v5",
    "v3_v4",
    "v3_v5",
    "v4_v5",
]

tasks = []


PROMPT_TEMPLATE_STARTER_CODE = dedent("""
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question_content}

{starter_code}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

```python
# YOUR CODE HERE
```
""")

PROMPT_TEMPLATE_NO_STARTER_CODE = dedent("""
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question_content}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

```python
# YOUR CODE HERE
```
""")


@scorer(metrics=[accuracy()])
def lcb_scorer():
    async def score(state: TaskState, target: Target):
        # Extract model output text
        model_answer = state.output.completion
        # Extract code block from model output
        generated_code = extract_code(model_answer)

        # Build single-sample spec from target metadata
        inputs = state.metadata.get("inputs")
        outputs = state.metadata.get("outputs")
        fn_name = state.metadata.get("fn_name")

        # inputs = json.loads(inputs)
        # outputs = json.loads(outputs)

        print(f"Inputs: {inputs}")
        print(f"Outputs: {outputs}")
        print(f"Fn name: {fn_name}")
        print(f"Generated code: {generated_code}")

        passed = run_test(
            inputs=inputs,
            outputs=outputs,
            fn_name=fn_name,
            model_answer=generated_code,
            timeout=3,
        )

        print(f"Passed: {passed}")

        passed = 1

        return Score(value=float(passed))

    return score


def record_to_sample(record):
    if starter_code := record.get("starter_code", None):
        input = PROMPT_TEMPLATE_STARTER_CODE.format(
            question_content=record["question_content"], starter_code=starter_code
        )
    else:
        input = PROMPT_TEMPLATE_NO_STARTER_CODE.format(question_content=record["question_content"])

    public_test_cases = json.loads(record["public_test_cases"])
    private_test_cases = translate_private_test_cases(record["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]

    return Sample(
        input=input,
        target="",
        metadata={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(record["metadata"]).get("func_name", None),
        },
    )


for subset in configs:
    # To keep the base subset as the default, the others are named "lcb:codegeneration_v4", "lcb:codegeneration_v5"... etc
    name = "lcb:codegeneration" if subset == "v4_v5" else f"lcb:codegeneration_{subset}"
    task = LightevalTaskConfig(
        name=name,
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="lighteval/code_generation_lite",
        hf_subset=subset,  # https://github.com/LiveCodeBench/LiveCodeBench/tree/main?tab=readme-ov-file#dataset-versions
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=32768,
        metrics=[Metrics.lcb_codegen_metric],
        stop_sequence=[],  # no stop sequence, will use EOS token
        version=0,
        sample_fields=record_to_sample,
        solver=[generate(cache=True)],
        scorer=lcb_scorer(),
        sandbox="docker",
        epochs=4,
        epochs_reducer="pass_at_1",
    )
    tasks.append(task)


TASKS_TABLE = tasks

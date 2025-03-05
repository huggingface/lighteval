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
import json

from lighteval.utils.code.types import CodeRunnerOutput
from lighteval.utils.code.utils import prepare_fn
from lighteval.utils.imports import is_e2b_available


# STDIO is related to a code snippet that reads from stdin and writes to stdout
EVALUATION_SCRIPT_STDIO = """\
import subprocess
import json

def evaluate_code(code, test_cases):
    outputs = []
    exec_timeout = 5
    for test_case in test_cases:
        process = subprocess.run(
            ["python3", "-c", code],
            input=test_case,
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )
        if process.returncode != 0:  # Error in execution
            outputs.append('')
        outputs.append(process.stdout.strip())
    return outputs

evaluate_code({code}, json.loads({test_cases}))
"""

# CALL_BASED is related to a function that is executed, and the output printed (so it can be captured)
# No input needs to be passed, the arguments will be passed to a function call internally (has to be
# present in the code senippet supplied).
EVALUATION_SCRIPT_CALL_BASED = """\
import subprocess
import json
import tempfile
import os

def evaluate_code(code, test_cases):
    outputs = []
    exec_timeout = 5

    for test_case in test_cases:
        script = code.format(args=test_case)
        # Create a temporary Python file with just the code
        try:
            process = subprocess.run(
                ["python", "-c", script],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:
                outputs.append("Execution error: " + process.stderr.strip())
            else:
                outputs.append(process.stdout.strip())

        except subprocess.TimeoutExpired:
            outputs.append("Timeout error")

    return outputs

evaluate_code({code}, json.loads({test_cases}))
"""


# TODO: Update the inputs type, it should be a list with dicts, and a list with
# the inputs inside, so we can gather the data there, i.e.
# [{input: [input1, input2, ...], "fn_name": ..}, {input: [input1, input2, ...], "fn_name": ..}, ...]
def code_runner_e2b(inputs: list[list[str]], code: list[str], timeout: int = 300, **kwargs) -> CodeRunnerOutput:
    """Runs the code in an e2b sandbox and returns the output or error: https://e2b.dev/

    Args:
        inputs (list[list[str]]): List of lists with test cases, each list corresponding
            one element in the code list.
        code (list[str]): The list of code snippets to run. For example they could correspond to N
            generations of a model.
        timeout (int): The timeout in seconds for the code to run. Defaults to 300.
        **kwargs: Additional keyword arguments. This arguments will be passed to the e2b sandbox.
            For example, `language` can be passed to specify the language of the code (defaults
            to "python"), or the `request_timeout` (defaults to 3 seconds).

    Returns:
        CodeRunnerOutput: The output or error of the code.
    """
    if is_e2b_available():
        from dotenv import load_dotenv
        from e2b_code_interpreter import Sandbox

        load_dotenv()
    else:
        raise ValueError("e2b is not available, install it with `pip install e2b_code_interpreter`")

    template = kwargs.get("template", "stdio")
    if template == "stdio":

        def populate_template(code: str, info: list[str]) -> str:
            return EVALUATION_SCRIPT_STDIO.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info)))

    elif template == "call_based":
        # This template doesn't take inputs, they must be written in the code snippet directly.
        # The code snippets have to be prepared beforehand, as we won't pass them in the template
        # script.
        # In this case we prepare the function by adding a call to the function generated
        # (it shouldn't be present). After that, the function is added to the template
        def populate_template(code: str, info: list[str]) -> str:
            formatted_code = prepare_fn(code, fn_name=kwargs.get("fn_name"))

            return EVALUATION_SCRIPT_CALL_BASED.format(
                code=json.dumps(formatted_code), test_cases=json.dumps(json.dumps(info))
            )

    else:
        raise ValueError(f"Template {template} is not valid")

    # Prepare the "scripts" to run
    scripts = []
    for code_snippet, info in zip(code, inputs):
        scripts.append(populate_template(code_snippet, info))

    language = kwargs.get("language", "python")
    outputs = []
    with Sandbox(timeout=timeout, request_timeout=kwargs.get("request_timeout", 3)) as sbx:
        for script in scripts:
            execution = sbx.run_code(script, language=language)
            # The execution returns an object like the following:
            # Execution(Results: [Result(...)], Logs: Logs(stdout: [], stderr: []), Error: None)
            # Get the text representation of the result
            # If everything went well, the result will be the output of the code in the .text attribute
            outputs.append(execution.text)

    return CodeRunnerOutput(output=outputs, error=None)

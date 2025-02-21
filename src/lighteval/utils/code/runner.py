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
from lighteval.utils.code.e2b import code_runner_e2b
from lighteval.utils.code.local import code_runner_local
from lighteval.utils.code.types import CodeRunnerOutput


def code_runner(inputs: list[list[str]], code: list[str], timeout: int = 300, **kwargs) -> CodeRunnerOutput:
    """Runs the code with the given input and returns the output or error.

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
    sandbox = kwargs.get("sandbox", None)
    if sandbox == "e2b":
        return code_runner_e2b(inputs, code, timeout, **kwargs)

    return code_runner_local(inputs, code, timeout, **kwargs)

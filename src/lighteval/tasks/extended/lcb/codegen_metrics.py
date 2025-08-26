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
"""This module contains helper functions copied and modified from
https://github.com/LiveCodeBench/LiveCodeBench
and
https://github.com/QwenLM/Qwen2.5-Coder/tree/main/qwencoder-eval/instruct/livecode_bench
"""

import ast
import base64
import faulthandler
import json
import multiprocessing
import os
import pickle
import platform

# to run the solution files we're using a timing based approach
import signal
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from typing import Callable, Optional

# used for testing the code that reads from input
from unittest.mock import mock_open, patch

import numpy as np
from tqdm import tqdm


sys.set_int_max_str_digits(50000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occurred: alarm went off")
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except Exception:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code


def call_method(method: Callable, inputs: list | str):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str) -> str:  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return


def compile_code(code: str, timeout: int) -> ModuleType:
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accessible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str) -> list[str]:
    # you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int) -> list[int | bool]:
    # call-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        # The code could not be compiled as the text was maybe malformed, return [-4] error code.
        return [-4]

    method = get_function(compiled_sol, fn_name)
    if method is None:
        # The function to evaluate could not be extracted from the code, return [-4] error code.
        return [-4]

    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]

    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            all_results.append(tmp_result)

            if not tmp_result:
                return all_results

        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results

            else:
                all_results.append(-4)
                return all_results

        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results


def grade_stdio(  # noqa: C901
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
) -> list[int | bool]:
    # runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    # we wrap the given code inside another function
    code = make_function(code)

    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        # The code could not be compiled as the text was maybe malformed, return [-4] error code.
        return [-4]

    method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        # The function to evaluate could not be extracted from the code, return [-4] error code.
        return [-4]

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results

                else:
                    all_results.append(-4)
                    return all_results

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            return all_results

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            # CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            # CASE 2: element-wise comparison
            # if there are floating elements
            # use `decimal` library for good floating point comparison
            # otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            # note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
            if not success:
                all_results.append(-2)
                return all_results

            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                all_results.append(-2)
                return all_results

            if decimal_prediction_line == decimal_gtout_line:
                continue

            all_results.append(-2)
            return all_results
        all_results.append(True)

    return all_results


def run_test(sample: dict[str, str], test=None, timeout: int = 6) -> list[int | bool]:
    """If test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    The return codes are all for errors. A True value means the function worked, and a False
    value means the function did not work.
    """
    signal.signal(signal.SIGALRM, timeout_handler)

    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB
    reliability_guard()

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError:
        in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None

        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if test is None:
        # assert False, "should not happen: test code is none"
        return in_outs

    if which_type == CODE_TYPE.call_based:
        signal.alarm(timeout)
        try:
            return grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=method_name,
                timeout=timeout,
            )

        except Exception:
            return [-4]

        finally:
            signal.alarm(0)

    elif which_type == CODE_TYPE.standard_input:
        signal.alarm(timeout)
        try:
            return grade_stdio(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                timeout=timeout,
            )

        except Exception:
            return [-4]

        finally:
            signal.alarm(0)

    return [-4]


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def check_correctness(sample, generation, timeout: int) -> list[int | bool]:
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`.
    """

    def _temp_run(sample, generation, result):
        result.append(run_test(sample, test=generation, timeout=timeout))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, result))
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]

    return result[0]


def evaluate_generations_by_problem(args: list[list]) -> list:
    problem_generations: list[str] = args[0]
    sample = args[1]
    timeout: int = args[2]

    res = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res = check_correctness(sample, o, timeout=timeout)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed

        except Exception as e:
            # The compilation failed, keep going
            pass

        finally:
            assert isinstance(curr_res, list), curr_res
            res.append(curr_res)

    return res


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    num_process_evaluate: int = 16,
    timeout=6,
) -> dict[int, list[int | bool]]:
    """We take the list of code generations and try to compile them
    and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset

    inputs = [
        [(generations_list[index], samples_list[index], timeout), index] for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(max_workers=num_process_evaluate) as executor:
            futures = {executor.submit(evaluate_generations_by_problem, arg): index for arg, index in inputs}

            results = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)} {results=}"

    return results


def codegen_metrics(
    samples,
    generations,
    k_list=[1, 5],
    num_process_evaluate=16,
    timeout=6,
):
    results = evaluate_generations(
        samples,
        generations,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    metrics = compute_metrics_from_results(results, k_list=k_list)

    return metrics, results


def estimate_pass_at_k(num_samples: np.ndarray, num_correct: np.ndarray, k: int) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def compute_metrics_from_results(results, k_list: list[int] = [1, 5]) -> dict[str, np.ndarray]:
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in k_list if (total >= k).all()
    }
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in k_list if (total >= k).all()}
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


def translate_private_test_cases(encoded_data: str) -> dict[str, str]:
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def extract_code(model_output: str) -> str:
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

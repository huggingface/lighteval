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

"""This file should be launched using `python -m pytest script_name.py`. It must stay at the same level or above as main"""
import os

import pytest
from pytest import approx

from lighteval.main_accelerate import main  # noqa: E402
from run_evals_accelerate import get_parser
from tests.reference_scores.reference_task_scores import RESULTS_FULL, RESULTS_LITE  # noqa: E402
from tests.reference_scores.reference_tasks import ALL_SUBSETS


# Set env var for deterministic run of models
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Set cache for github actions
os.environ["HF_DATASETS_CACHE"] = "cache/datasets/"
os.environ["HF_HOME"] = "cache/models/"

# To add new models or tasks, change here
# ! The correct results must be present in reference_task_scores
MODELS = ["gpt2"]
TASKS = ALL_SUBSETS
FULL_TEST = os.environ.get("LIGHTEVAL_FULL_TEST", False)


def run_model_predictions_full(model: str, tasks: list):
    """Runs the full main as a black box, using the input model and tasks, on all samples without parallelism"""
    lighteval_args = ["--model_args", f"pretrained={model}", "--tasks", ",".join(tasks)]
    lighteval_args += [
        "--override_batch_size",
        "1",
        "--output_dir",
        "",
        "--dataset_loading_processes",
        "1",
        "--save_details",
    ]
    parser = get_parser()
    args = parser.parse_args(lighteval_args)
    results = main(args)
    return results


def run_model_predictions_lite(model: str, tasks: list):
    """Runs the full main as a black box, using the input model and tasks, on 10 samples without parallelism"""
    lighteval_args = ["--model_args", f"pretrained={model}", "--tasks", ",".join(tasks)]
    lighteval_args += [
        "--override_batch_size",
        "1",
        "--output_dir",
        "",
        "--dataset_loading_processes",
        "1",
        "--save_details",
    ]
    lighteval_args += ["--max_samples", "10"]
    parser = get_parser()
    args = parser.parse_args(lighteval_args)
    results = main(args)
    return results


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Initializes the main test setup. This function is automatically called by pytest and
    should not be called manually.

    Every function with "model_input" as arguments will be sent the "parameters".
    This function will be run only once, ensuring that each model is run only once on the selected tasks.
    (This is better than using fixtures as fixtures are re-run once for each test, which is not a behavior we want).
    """
    parameters = []

    # If model_input is a test function argument
    # (= the function requires a fixture)
    if "model_input" in metafunc.fixturenames:
        tasks = TASKS  # must be a list not a file name
        for model in MODELS:
            if FULL_TEST:
                predictions_full = run_model_predictions_full(model, tasks)
                for eval_name in tasks:
                    for metric, reference in RESULTS_FULL[model][eval_name].items():
                        if len(eval_name.split("|")) == 4:
                            eval_name = "|".join(eval_name.split("|")[:-1])
                        prediction = predictions_full["results"][eval_name.replace("|", ":")][metric]
                        parameters.append((model, "all", eval_name, metric, prediction, reference))
            else:
                predictions_lite = run_model_predictions_lite(model, tasks)
                for eval_name in tasks:
                    for metric, reference in RESULTS_LITE[model][eval_name].items():
                        if len(eval_name.split("|")) == 4:
                            eval_name = "|".join(eval_name.split("|")[:-1])
                        prediction = predictions_lite["results"][eval_name.replace("|", ":")][metric]
                        parameters.append((model, "lite", eval_name, metric, prediction, reference))
        metafunc.parametrize("model_input", parameters, scope="session")


def test_model_prediction(model_input: tuple):
    """Evaluates a model on a full task - is parametrized using pytest_generate_test"""
    model_name, test_type, eval_name, metric, source, prediction = model_input
    assert source == approx(
        prediction, rel=1e-4
    ), f"Model {model_name} on {test_type} samples, for eval {eval_name}, metric {metric} incorrect"


if __name__ == "__main__":
    parameters = []
    tasks = TASKS
    for model in MODELS:
        if FULL_TEST:
            predictions_full = run_model_predictions_full(model, tasks)
            for eval_name in tasks:
                for metric, reference in RESULTS_FULL[model][eval_name].items():
                    prediction = predictions_full["results"][eval_name.replace("|", ":")][metric]
                    parameters.append((model, "all", eval_name, metric, prediction, reference))
        else:
            predictions_lite = run_model_predictions_lite(model, tasks)
            for eval_name in tasks:
                for metric, reference in RESULTS_LITE[model][eval_name].items():
                    prediction = predictions_lite["results"][eval_name.replace("|", ":")][metric]
                    parameters.append((model, "lite", eval_name, metric, prediction, reference))
    print(parameters)

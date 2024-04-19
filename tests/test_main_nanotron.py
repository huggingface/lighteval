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

"""This file should be launched using `pytest tests/test_main_nanotron.py -sv`. It must stay at the same level or above as main"""
import json
import os

import pytest
from pytest import approx

from lighteval.main_nanotron import main  # noqa: E402
from run_evals_nanotron import get_parser
from tests.reference_scores.reference_task_scores_nanotron import RESULTS_NANOTRON_FULL, RESULTS_NANOTRON_LITE


# Set env var for deterministic run of models
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Set cache for github actions
os.environ["HF_DATASETS_CACHE"] = "cache/datasets/"
os.environ["HF_HOME"] = "cache/models/"

# To add new models or tasks, change here
# ! The correct results must be present in reference_task_scores
MODELS = [{"name": "LLama-119M", "config_path": "/fsx/haojun/lighteval_evaluation_model/config.yaml"}]
LIGHTEVAL_CONFIG_PATH = "/fsx/haojun/lighteval/tests/config/lighteval_config_override_custom.yaml"  # define tasks
SAVE_RESULTS = (
    False  # whether you want to save the results in json format, and update reference_tasks_scores_nanotron.py later
)
RESULTS_DIRECTORY = "/fsx/haojun/lighteval/tests"
FULL_TEST = os.environ.get("LIGHTEVAL_FULL_TEST", False)  # Full evaluation or Lite evaluation

# set env variables as nanotron need them
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29400"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"


def run_model_predictions_full(config_path: str, lighteval_config_path: str):
    """Runs the full main as a black box, using the input model and tasks, on all samples without parallelism"""
    lighteval_args = ["--checkpoint-config-path", f"{config_path}", "--lighteval-override", f"{lighteval_config_path}"]
    lighteval_args += ["--max_samples", "10000000"]
    parser = get_parser()
    args = parser.parse_args(lighteval_args)
    results = main(args.checkpoint_config_path, args=args)
    return results


def run_model_predictions_lite(config_path: str, lighteval_config_path: str):
    """Runs the full main as a black box, using the input model and tasks, on 10 samples without parallelism"""
    lighteval_args = ["--checkpoint-config-path", f"{config_path}", "--lighteval-override", f"{lighteval_config_path}"]
    lighteval_args += ["--max_samples", "4"]
    parser = get_parser()
    args = parser.parse_args(lighteval_args)
    results = main(args.checkpoint_config_path, args=args)
    return results


def generate_full_test_parameters(model, tasks, results_nanotron_full):
    predictions_full = run_model_predictions_full(model["config_path"], LIGHTEVAL_CONFIG_PATH)
    if SAVE_RESULTS:
        with open(f"{RESULTS_DIRECTORY}/predictions_full.json", "w") as file:
            json.dump(predictions_full["results"], file, indent=4)

    parameters = []
    for eval_name in tasks:
        for metric, reference in results_nanotron_full[model["name"]][eval_name].items():
            if len(eval_name.split("|")) == 4:
                eval_name = "|".join(eval_name.split("|")[:-1])
            prediction = predictions_full["results"][eval_name.replace("|", ":")][metric]
            parameters.append((model, "all", eval_name, metric, prediction, reference))
    return parameters


def generate_lite_test_parameters(model, tasks, results_nanotron_lite):
    predictions_lite = run_model_predictions_lite(model["config_path"], LIGHTEVAL_CONFIG_PATH)
    if SAVE_RESULTS:
        with open(f"{RESULTS_DIRECTORY}/predictions_lite.json", "w") as file:
            json.dump(predictions_lite["results"], file, indent=4)

    parameters = []
    for eval_name in tasks:
        for metric, reference in results_nanotron_lite[model["name"]][eval_name].items():
            if len(eval_name.split("|")) == 4:
                eval_name = "|".join(eval_name.split("|")[:-1])
            prediction = predictions_lite["results"][eval_name.replace("|", ":")][metric]
            parameters.append((model, "lite", eval_name, metric, prediction, reference))
    return parameters


def pytest_generate_tests(metafunc: pytest.Metafunc):
    parameters = []

    if "model_input" in metafunc.fixturenames:
        for model in MODELS:
            if FULL_TEST:
                tasks = list(RESULTS_NANOTRON_FULL[model["name"]].keys())
                parameters.extend(generate_full_test_parameters(model, tasks, RESULTS_NANOTRON_FULL))
            else:
                tasks = list(RESULTS_NANOTRON_LITE[model["name"]].keys())
                parameters.extend(generate_lite_test_parameters(model, tasks, RESULTS_NANOTRON_LITE))

    metafunc.parametrize("model_input", parameters, scope="session")


def test_model_prediction(model_input: tuple):
    """Evaluates a model on a full task - is parametrized using pytest_generate_test"""
    model_name, test_type, eval_name, metric, prediction, source = model_input
    assert source == approx(
        prediction, rel=1e-4
    ), f"Model {model_name} on {test_type} samples, for eval {eval_name}, metric {metric} incorrect"

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
from functools import lru_cache, partial
from typing import Callable, List, Literal, Tuple

import pytest
from pytest import approx

from lighteval.main_accelerate import main  # noqa: E402
from lighteval.parsers import parser_accelerate
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
ModelInput = Tuple[str, str, str, str, Callable[[], dict], float]


# Caching here to avoid re-running predictions for every single test, the size should be >= MODELS
@lru_cache(maxsize=len(MODELS))
def run_model_predictions_full(model: str, tasks: tuple):
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
    parser = parser_accelerate()
    args = parser.parse_args(lighteval_args)
    results = main(args)
    return results


@lru_cache(maxsize=len(MODELS))
def run_model_predictions_lite(model: str, tasks: tuple):
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
    parser = parser_accelerate()
    args = parser.parse_args(lighteval_args)
    results = main(args)
    return results


def generate_test_parameters(tasks: List[str]) -> List[ModelInput]:
    """Generate test parameters for all models and tasks."""

    def generate_model_parameters(
        model: str, test_type: Literal["full", "lite"], prediction_func: Callable
    ) -> List[ModelInput]:
        results = RESULTS_FULL if test_type == "full" else RESULTS_LITE
        return [
            (model, test_type, normalize_eval_name(eval_name), metric, prediction_func, reference)
            for eval_name in tasks
            for metric, reference in results[model][eval_name].items()
        ]

    parameters = []
    for model in MODELS:
        if FULL_TEST:
            # Don't call the function during collection!! Very expensive
            predictions_full = partial(run_model_predictions_full, model, tuple(tasks))
            parameters.extend(generate_model_parameters(model, "full", predictions_full))
        else:
            predictions_lite = partial(run_model_predictions_lite, model, tuple(tasks))
            parameters.extend(generate_model_parameters(model, "lite", predictions_lite))

    return parameters


def normalize_eval_name(eval_name: str) -> str:
    """Normalize evaluation name by removing the last part if it has 4 components."""
    parts = eval_name.split("|")
    return "|".join(parts[:3]) if len(parts) == 4 else eval_name


# generates the model predictions parameters at test collection time
parameters: list[ModelInput] = generate_test_parameters(TASKS)
ids = [f"{model_input[0]}_{model_input[1]}_{model_input[2]}_{model_input[3]}" for model_input in parameters]


@pytest.mark.parametrize("model_input", parameters, ids=ids)
def test_model_prediction(model_input: ModelInput):
    """Evaluates a model on a full task - is parametrized using pytest_generate_test"""
    model_name, test_type, eval_name, metric, get_predictions, reference = model_input
    prediction = get_predictions()["results"][eval_name.replace("|", ":")][metric]
    assert reference == approx(
        prediction, rel=1e-4
    ), f"Model {model_name} on {test_type} samples, for eval {eval_name}, metric {metric} incorrect"


if __name__ == "__main__":
    parameters = generate_test_parameters(TASKS)
    print(parameters)

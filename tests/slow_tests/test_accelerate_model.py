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

import json
import os
from functools import lru_cache, partial
from typing import Callable, Tuple

import pytest
from deepdiff import DeepDiff

from lighteval.main_accelerate import accelerate  # noqa: E402


# Set env var for deterministic run of models
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

MODELS_ARGS = [
    # {"model_name": "gpt2", "use_chat_template": False, "revision": "main", "results_file": "tests/reference_scores/gpt2-results.json"},
    {
        "model_name": "examples/model_configs/transformers_model.yaml",
        "use_chat_template": True,
        "results_file": "tests/reference_scores/SmolLM2-1.7B-Instruct-results-accelerate.json",
    }
]
TASKS_PATH = "examples/test_tasks.txt"
CUSTOM_TASKS_PATH = "examples/custom_tasks_tests.py"

ModelInput = Tuple[str, Callable[[], dict]]


@lru_cache(maxsize=len(MODELS_ARGS))
def run_model(model_name: str, use_chat_template: bool):
    """Runs the full main as a black box, using the input model and tasks, on 10 samples without parallelism"""
    results = accelerate(
        model_args=model_name,
        tasks=TASKS_PATH,
        use_chat_template=use_chat_template,
        output_dir="",
        dataset_loading_processes=1,
        save_details=False,
        max_samples=10,
        custom_tasks=CUSTOM_TASKS_PATH,
    )
    return results


def generate_tests() -> list[ModelInput]:
    """Generate test parameters for all models and tasks."""

    tests = []
    for model_args in MODELS_ARGS:
        predictions_lite = partial(run_model, model_args["model_name"], model_args["use_chat_template"])
        tests.append((model_args, predictions_lite))

    return tests


# generates the model predictions parameters at test collection time
tests: list[ModelInput] = generate_tests()
ids = [f"{model_input[0]['model_name']}" for model_input in tests]


@pytest.mark.slow
@pytest.mark.parametrize("tests", tests, ids=ids)
def test_accelerate_model_prediction(tests: list[ModelInput]):
    """Evaluates a model on a full task - is parametrized using pytest_generate_test"""
    model_args, get_predictions = tests

    # Load the reference results
    with open(model_args["results_file"], "r") as f:
        reference_results = json.load(f)["results"]

    # Change the key names, replace '|' with ':'
    reference_results = {k.replace("|", ":"): v for k, v in reference_results.items()}

    # Get the predictions
    predictions = get_predictions()["results"]

    # Convert defaultdict values to regular dict for comparison
    predictions_dict = {k: dict(v) if hasattr(v, "default_factory") else v for k, v in predictions.items()}

    # Compare the predictions with the reference results
    diff = DeepDiff(reference_results, predictions_dict, ignore_numeric_type_changes=True, math_epsilon=0.05)

    assert diff == {}, f"Differences found: {diff}"

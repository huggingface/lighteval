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

import nanotron.constants as nanotron_constants  # Add this import
import pytest
import yaml
from deepdiff import DeepDiff
from huggingface_hub import snapshot_download
from packaging.version import Version

from lighteval.main_nanotron import nanotron  # noqa: E402


# Set env var for deterministic run of models
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# Download the model checkpoint
@pytest.fixture(scope="session", autouse=True)
def download_model():
    snapshot_download(
        repo_id="HuggingFaceTB/SmolLM2-nanotron-ckpt",
        allow_patterns=["1700M/final/*"],
        local_dir="./SmolLM2-nanotron-ckpt/",
    )


MODELS_ARGS = [
    # {"model_name": "gpt2", "use_chat_template": False, "revision": "main", "results_file": "tests/reference_scores/gpt2-results.json"},
    {
        "model_name": "SmolLM2-nanotron-ckpt/1700M/final/config.yaml",
        "lighteval_config_path": "examples/lighteval_config_override_nanotron_tests.yaml",
        "results_file": "tests/reference_scores/SmolLM2-1.7B-Instruct-results-nanotron.json",
    }
]
TASKS_PATH = "examples/test_tasks.txt"
CUSTOM_TASKS_PATH = "examples/custom_tasks_tests.py"

ModelInput = Tuple[str, Callable[[], dict]]


# Set data_stages to null in config.yaml before running tests
def set_data_stages_to_null(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    keys_to_keep = ["model", "tokenizer", "general", "parallelism"]
    keys_to_delete = [key for key in config.keys() if key not in keys_to_keep]
    for key in keys_to_delete:
        del config[key]
    if "parallelism" in config and config["parallelism"] is not None:
        if "tp_recompute_allgather" in config["parallelism"]:
            del config["parallelism"]["tp_recompute_allgather"]
        if "recompute_layer" in config["parallelism"]:
            del config["parallelism"]["recompute_layer"]
    if "model" in config and config["model"] is not None:
        if "model_config" in config["model"]:
            if "rope_theta" in config["model"]["model_config"]:
                del config["model"]["model_config"]["rope_theta"]
            if "rope_interleaved" in config["model"]["model_config"]:
                del config["model"]["model_config"]["rope_interleaved"]
    # config["data_stages"] = None
    # if "checkpoints" in config and config["checkpoints"] is not None:
    #     if "save_final_state" in config["checkpoints"]:
    #         del config["checkpoints"]["save_final_state"]
    # if "optimizer" in config and config["optimizer"] is not None:
    #     if "optimizer_factory" in config["optimizer"]:
    #         del config["optimizer"]["optimizer_factory"]
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


@lru_cache(maxsize=len(MODELS_ARGS))
def run_model(checkpoint_config_path: str, lighteval_config_path: str):
    """Runs the full main as a black box, using the input model and tasks, on 10 samples without parallelism"""
    # Emulate torchrun launch
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "60000"  # Or any other free port
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    results = nanotron(
        checkpoint_config_path=checkpoint_config_path,
        lighteval_config_path=lighteval_config_path,
    )
    return results


def generate_tests() -> list[ModelInput]:
    """Generate test parameters for all models and tasks."""

    tests = []
    for model_args in MODELS_ARGS:
        predictions_lite = partial(run_model, model_args["model_name"], model_args["lighteval_config_path"])
        tests.append((model_args, predictions_lite))

    return tests


# generates the model predictions parameters at test collection time
tests: list[ModelInput] = generate_tests()
ids = [f"{model_input[0]['model_name']}" for model_input in tests]


@pytest.mark.slow
@pytest.mark.parametrize("tests", tests, ids=ids)
def test_nanotron_model(tests: list[ModelInput], monkeypatch):  # Add monkeypatch fixture
    """Evaluates a model on a full task - is parametrized using pytest_generate_test"""
    model_args, get_predictions = tests

    # Set data_stages to null in config.yaml before running tests
    set_data_stages_to_null(model_args["model_name"])

    # Monkeypatch CHECKPOINT_VERSION to bypass version check
    monkeypatch.setattr(nanotron_constants, "CHECKPOINT_VERSION", Version("1.4"))

    predictions = get_predictions()["results"]

    # Load the reference results
    with open(model_args["results_file"], "r") as f:
        reference_results = json.load(f)["results"]

    # Change the key names, replace '|' with ':'
    reference_results = {k.replace("|", ":"): v for k, v in reference_results.items()}

    # Convert defaultdict values to regular dict for comparison
    predictions_dict = {k: dict(v) if hasattr(v, "default_factory") else v for k, v in predictions.items()}

    diff = DeepDiff(reference_results, predictions_dict, ignore_numeric_type_changes=True, math_epsilon=0.05)

    assert diff == {}, f"Differences found: {diff}"

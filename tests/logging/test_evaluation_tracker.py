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
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from datasets import Dataset
from huggingface_hub import HfApi

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.info_loggers import DetailsLogger

# ruff: noqa
from tests.fixtures import TESTING_EMPTY_HF_ORG_ID, testing_empty_hf_org_id


@pytest.fixture
def mock_evaluation_tracker(request):
    passed_params = {}
    if request.keywords.get("evaluation_tracker"):
        passed_params = request.keywords["evaluation_tracker"].kwargs

    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs = {
            "output_dir": temp_dir,
            "save_details": passed_params.get("save_details", False),
            "push_to_hub": passed_params.get("push_to_hub", False),
            "push_to_tensorboard": passed_params.get("push_to_tensorboard", False),
            "hub_results_org": passed_params.get("hub_results_org", ""),
        }
        tracker = EvaluationTracker(**kwargs)
        tracker.general_config_logger.model_name = "test_model"
        yield tracker


@pytest.fixture
def mock_datetime(monkeypatch):
    mock_date = datetime(2023, 1, 1, 12, 0, 0)

    class MockDatetime:
        @classmethod
        def now(cls):
            return mock_date

        @classmethod
        def fromisoformat(cls, date_string: str):
            return mock_date

    monkeypatch.setattr("lighteval.logging.evaluation_tracker.datetime", MockDatetime)
    return mock_date


def test_results_logging(mock_evaluation_tracker: EvaluationTracker):
    task_metrics = {
        "task1": {"accuracy": 0.8, "f1": 0.75},
        "task2": {"precision": 0.9, "recall": 0.85},
    }
    mock_evaluation_tracker.metrics_logger.metric_aggregated = task_metrics

    mock_evaluation_tracker.save()

    results_dir = Path(mock_evaluation_tracker.output_dir) / "results" / "test_model"
    assert results_dir.exists()

    result_files = list(results_dir.glob("results_*.json"))
    assert len(result_files) == 1

    with open(result_files[0], "r") as f:
        saved_results = json.load(f)

    assert "results" in saved_results
    assert saved_results["results"] == task_metrics
    assert saved_results["config_general"]["model_name"] == "test_model"


@pytest.mark.evaluation_tracker(save_details=True)
def test_details_logging(mock_evaluation_tracker, mock_datetime):
    task_details = {
        "task1": [DetailsLogger.CompiledDetail(hashes=None, truncated=10, padded=5)],
        "task2": [DetailsLogger.CompiledDetail(hashes=None, truncated=20, padded=10)],
    }
    mock_evaluation_tracker.details_logger.details = task_details

    mock_evaluation_tracker.save()

    date_id = mock_datetime.isoformat().replace(":", "-")
    details_dir = Path(mock_evaluation_tracker.output_dir) / "details" / "test_model" / date_id
    assert details_dir.exists()

    for task in ["task1", "task2"]:
        file_path = details_dir / f"details_{task}_{date_id}.parquet"
        dataset = Dataset.from_parquet(str(file_path))
        assert len(dataset) == 1
        assert int(dataset[0]["truncated"]) == task_details[task][0].truncated
        assert int(dataset[0]["padded"]) == task_details[task][0].padded


@pytest.mark.evaluation_tracker(save_details=False)
def test_no_details_output(mock_evaluation_tracker: EvaluationTracker):
    mock_evaluation_tracker.save()

    details_dir = Path(mock_evaluation_tracker.output_dir) / "details" / "test_model"
    assert not details_dir.exists()


@pytest.mark.skip(  # skipif
    reason="Secrets are not available in this environment",
    # condition=os.getenv("HF_TEST_TOKEN") is None,
)
@pytest.mark.evaluation_tracker(push_to_hub=True, hub_results_org=TESTING_EMPTY_HF_ORG_ID)
def test_push_to_hub_works(testing_empty_hf_org_id, mock_evaluation_tracker: EvaluationTracker, mock_datetime):
    # Prepare the dummy data
    task_metrics = {
        "task1": {"accuracy": 0.8, "f1": 0.75},
        "task2": {"precision": 0.9, "recall": 0.85},
    }
    mock_evaluation_tracker.metrics_logger.metric_aggregated = task_metrics

    task_details = {
        "task1": [DetailsLogger.CompiledDetail(truncated=10, padded=5)],
        "task2": [DetailsLogger.CompiledDetail(truncated=20, padded=10)],
    }
    mock_evaluation_tracker.details_logger.details = task_details
    mock_evaluation_tracker.save()

    # Verify using HfApi
    api = HfApi()

    # Check if repo exists and it's private
    expected_repo_id = f"{testing_empty_hf_org_id}/details_test_model_private"
    assert api.repo_exists(repo_id=expected_repo_id, repo_type="dataset")
    assert api.repo_info(repo_id=expected_repo_id, repo_type="dataset").private

    repo_files = api.list_repo_files(repo_id=expected_repo_id, repo_type="dataset")
    # Check if README.md exists
    assert any(file == "README.md" for file in repo_files)

    # Check that both results files were uploaded
    result_files = [file for file in repo_files if file.startswith("results_")]
    assert len(result_files) == 2
    assert len([file for file in result_files if file.endswith(".json")]) == 1
    assert len([file for file in result_files if file.endswith(".parquet")]) == 1

    # Check that the details dataset was uploaded
    details_files = [file for file in repo_files if "details_" in file and file.endswith(".parquet")]
    assert len(details_files) == 2

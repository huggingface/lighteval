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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.info_loggers import DetailsLogger


@pytest.fixture
def mock_evaluation_tracker():
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = EvaluationTracker(
            output_dir=temp_dir,
            save_results=True,
            save_details=True,
            save_tensorboard=True,
        )
        tracker.general_config_logger.model_name = "test_model"
        yield tracker


def test_tensorboard_logging(mock_evaluation_tracker):
    mock_evaluation_tracker.save_results = False
    mock_evaluation_tracker.save_details = False
    mock_evaluation_tracker.save_tensorboard = True

    mock_evaluation_tracker.metrics_logger.metric_aggregated = {
        "task1": {"accuracy": 0.8, "f1": 0.75},
        "task2": {"precision": 0.9, "recall": 0.85},
    }

    mock_evaluation_tracker.save()

    with open(
        Path(mock_evaluation_tracker.output_res.path) / "tensorboard" / "test_model" / "events.out.tfevents", "r"
    ) as f:
        content = f.read()
    # Check if SummaryWriter was called
    assert "SummaryWriter" in content, "SummaryWriter was not called"

    # Check if scalar values were added
    assert "add_scalar" in content, "Scalar values were not added"
    assert "task1/accuracy" in content, "task1/accuracy was not logged"
    assert "task1/f1" in content, "task1/f1 was not logged"
    assert "task2/precision" in content, "task2/precision was not logged"
    assert "task2/recall" in content, "task2/recall was not logged"

    # Check if SummaryWriter was called

    # Check if scalar values were added


def test_results_logging(mock_evaluation_tracker: EvaluationTracker):
    mock_evaluation_tracker.metrics_logger.log("task1", {"accuracy": 0.8, "f1": 0.75})
    mock_evaluation_tracker.metrics_logger.log("task2", {"precision": 0.9, "recall": 0.85})

    mock_evaluation_tracker.save()

    results_dir = Path(mock_evaluation_tracker.output_res.path) / "results" / "test_model"
    assert results_dir.exists()

    result_files = list(results_dir.glob("results_*.json"))
    assert len(result_files) == 1

    with open(result_files[0], "r") as f:
        saved_results = json.load(f)

    assert "results" in saved_results
    assert saved_results["results"] == mock_evaluation_tracker.metrics_logger.metric_aggregated


def test_details_logging(mock_evaluation_tracker):
    mock_evaluation_tracker.details_logger.details = {
        "task1": [DetailsLogger.CompiledDetail(task_name="task1", num_samples=100)],
        "task2": [DetailsLogger.CompiledDetail(task_name="task2", num_samples=200)],
    }

    mock_evaluation_tracker.save()

    details_dir = Path(mock_evaluation_tracker.output_res.path) / "details" / "test_model"
    assert details_dir.exists()

    detail_files = list(details_dir.glob("details_*.parquet"))
    assert len(detail_files) == 2

    for file in detail_files:
        dataset = Dataset.from_parquet(file)
        assert len(dataset) == 1
        assert "task_name" in dataset.column_names
        assert "num_samples" in dataset.column_names


@patch("lighteval.logging.evaluation_tracker.HfApi")
@patch("lighteval.logging.evaluation_tracker.DatasetCard")
def test_recreate_metadata_card(mock_dataset_card, mock_hf_api, mock_evaluation_tracker):
    mock_api_instance = MagicMock()
    mock_hf_api.return_value = mock_api_instance
    mock_api_instance.list_repo_files.return_value = [
        "results_2023-01-01T00-00-00.json",
        "details_task1_2023-01-01T00-00-00.parquet",
        "details_task2_2023-01-01T00-00-00.parquet",
    ]

    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = [{"results": {"task1": {"accuracy": 0.8}, "task2": {"precision": 0.9}}}]

    with patch("lighteval.logging.evaluation_tracker.load_dataset", return_value=mock_dataset):
        mock_evaluation_tracker.recreate_metadata_card("test/repo")

    mock_dataset_card.from_template.assert_called_once()
    mock_card = mock_dataset_card.from_template.return_value
    mock_card.push_to_hub.assert_called_once_with("test/repo", repo_type="dataset")

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

"""
Automated testing framework for LightEval metrics.

This module provides a simple way to test metrics by providing input/output pairs.
You can define test cases with expected inputs and outputs, and the framework will
automatically run them and verify the results.
"""

import copy
import json
import logging
from dataclasses import field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


class MetricTestCase(BaseModel):
    """A test case for a metric with input and expected output."""

    name: str
    metric_class: str
    metric_params: dict[str, Any] = field(default_factory=dict)
    doc: dict[str, Any]
    model_response: dict[str, Any]
    expected_output: dict[str, float]
    tolerance: float = 1e-2
    description: str | None = None


class CorpusLevelMetricTestCase(BaseModel):
    """A test case for a corpus level metric with input and expected output."""

    name: str
    metric_class: str
    metric_name: str
    metric_params: dict[str, Any] = field(default_factory=dict)
    docs: list[dict[str, Any]]
    model_responses: list[dict[str, Any]]
    expected_output: float
    tolerance: float = 1e-2
    description: str | None = None


class MetricTestSuite(BaseModel):
    """A collection of test cases for metrics."""

    name: str
    test_cases: list[MetricTestCase | CorpusLevelMetricTestCase]
    corpus_level: bool = False
    description: str | None = None


SKIPPED_METRICS = [
    "faithfulness",  # Need GPU to run
    "bert_score",  # Issue with the scoring function, int too big to convert
    "simpleqa_judge",  # Need to setup for compute costs
]


class AutomatedMetricTester:
    """Automated testing framework for LightEval metrics."""

    METRIC_CLASSES = {metric.name: metric.value for metric in Metrics if metric.name not in SKIPPED_METRICS}

    def __init__(self):
        self.test_results = []

    def create_doc_from_dict(self, doc_dict: dict[str, Any]) -> Doc:
        """Create a Doc object from a dictionary representation."""
        return Doc(
            query=doc_dict.get("query", ""),
            choices=doc_dict.get("choices", []),
            gold_index=doc_dict.get("gold_index", 0),
            task_name=doc_dict.get("task_name", "test"),
            specific=doc_dict.get("specific", {}),
        )

    def create_model_response_from_dict(self, response_dict: dict[str, Any]) -> ModelResponse:
        """Create a ModelResponse object from a dictionary representation."""
        return ModelResponse(
            text=response_dict.get("text", []),
            logprobs=response_dict.get("logprobs", []),
            output_tokens=response_dict.get("output_tokens", []),
            argmax_logits_eq_gold=response_dict.get("argmax_logits_eq_gold", []),
        )

    def instantiate_metric(self, metric_class: str, metric_params: dict[str, Any]):
        """Get a metric from the Metrics enum with the given parameters."""
        if metric_class not in self.METRIC_CLASSES:
            raise ValueError(f"Unknown metric class: {metric_class}")

        # Get the metric from the Metrics enum
        if metric_params != {}:
            metric = self.METRIC_CLASSES[metric_class]
            metric_enum_value = copy.deepcopy(metric)(metric_params)
        else:
            metric_enum_value = self.METRIC_CLASSES[metric_class]

        # The Metrics enum values are already instantiated, so we just return them
        # The metric_params are ignored for now since the Metrics enum values are pre-configured
        return metric_enum_value

    def run_test_case(self, test_case: MetricTestCase | CorpusLevelMetricTestCase) -> dict[str, Any]:
        """Run a single test case and return the result."""
        # Check if metric is available in METRIC_CLASSES
        if test_case.metric_class not in self.METRIC_CLASSES:
            return {
                "test_case": test_case.name,
                "success": True,  # Mark as success to skip
                "expected": test_case.expected_output,
                "actual": None,
                "error": None,
                "skipped": True,
                "skip_reason": f"Metric '{test_case.metric_class}' not available in METRIC_CLASSES",
            }

        # Get the metric from the Metrics enum
        metric = self.instantiate_metric(test_case.metric_class, test_case.metric_params)

        if isinstance(test_case, CorpusLevelMetricTestCase):
            docs = [self.create_doc_from_dict(doc) for doc in test_case.docs]
            model_responses = [
                self.create_model_response_from_dict(response) for response in test_case.model_responses
            ]
            aggregation_function = metric.get_corpus_aggregations()[metric.metric_name]
            outputs_per_sample = [
                metric.compute_sample(doc=doc, model_response=model_response)[test_case.metric_name]
                for doc, model_response in zip(docs, model_responses)
            ]
            actual_output = aggregation_function(outputs_per_sample)

            success = self._compare_dict_outputs(actual_output, test_case.expected_output, test_case.tolerance)

            return {
                "test_case": test_case.name,
                "success": success,
                "error": None,
                "skipped": False,
                "skip_reason": None,
                "actual": actual_output,
                "expected": test_case.expected_output,
            }

        doc = self.create_doc_from_dict(test_case.doc)
        model_response = self.create_model_response_from_dict(test_case.model_response)

        # Check if this is a batched metric
        if hasattr(metric, "batched_compute") and metric.batched_compute:
            # For batched metrics, we need to pass lists of docs and responses
            sample_params = {
                "docs": [doc],
                "responses": [model_response],
            }
        else:
            # For non-batched metrics, use individual doc and model_response
            sample_params = {
                "doc": doc,
                "model_response": model_response,
            }

        # Run the metric using the Metrics enum value
        actual_output = metric.compute_sample(**sample_params)

        # For batched metrics, extract the first result since we're only testing with one sample
        if hasattr(metric, "batched_compute") and metric.batched_compute and isinstance(actual_output, list):
            actual_output = actual_output[0]

        # Compare with expected output
        success = self._compare_dict_outputs(actual_output, test_case.expected_output, test_case.tolerance)
        return {
            "test_case": test_case.name,
            "success": success,
            "expected": test_case.expected_output,
            "actual": actual_output,
            "error": None,
            "skipped": False,
        }

    def _compare_scalar_outputs(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Compare scalar outputs with tolerance."""
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            # Use pytest.approx for float comparison
            return actual == pytest.approx(expected, abs=tolerance)
        return actual == expected

    def _compare_dict_outputs(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Compare outputs with tolerance. Handles both dict and scalar types."""
        # If either is not a dict, treat as scalar comparison
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return self._compare_scalar_outputs(actual, expected, tolerance)

        # Both are dicts, compare keys first
        if set(actual.keys()) != set(expected.keys()):
            return False

        # Compare each value
        for key in actual.keys():
            actual_value = actual[key]
            expected_value = expected[key]

            # Handle corpus metric inputs (objects with specific types)
            if hasattr(actual_value, "__class__") and "CorpusMetricInput" in str(actual_value.__class__):
                # For corpus metric inputs, just check that the key exists and the object is created
                continue
            elif hasattr(actual_value, "__class__") and "np.float64" in str(actual_value.__class__):
                # For numpy float64 values, convert to regular float for comparison
                actual_value = float(actual_value)

            if not self._compare_scalar_outputs(actual_value, expected_value, tolerance):
                return False

        return True

    def run_test_suite(self, test_suite: MetricTestSuite) -> list[dict[str, Any]]:
        """Run a complete test suite and return results."""
        logger.info(f"Running test suite: {test_suite.name}")
        if test_suite.description:
            logger.info(f"Description: {test_suite.description}")

        results = []
        for test_case in test_suite.test_cases:
            result = self.run_test_case(test_case)
            results.append(result)

            if result.get("skipped", False):
                logger.info(f"⏭ {test_case.name}: SKIPPED - {result.get('skip_reason', 'Unknown reason')}")
            elif result["success"]:
                logger.info(f"✓ {test_case.name}: PASSED")
            else:
                logger.error(f"✗ {test_case.name}: FAILED")
                if result["error"]:
                    logger.error(f"  Error: {result['error']}")
                else:
                    logger.error(f"  Expected: {result['expected']}")
                    logger.error(f"  Actual: {result['actual']}")

        return results

    def run_test_suites_from_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Run test suites from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Multiple test suites
            all_results = []
            for suite_data in data:
                test_suite = MetricTestSuite(**suite_data)
                results = self.run_test_suite(test_suite)
                all_results.extend(results)
            return all_results
        else:
            # Single test suite
            test_suite = MetricTestSuite(**data)
            return self.run_test_suite(test_suite)

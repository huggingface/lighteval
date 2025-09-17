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
Pytest integration for the automated metric testing framework.

This module provides pytest fixtures and test functions that can load and run
test cases from JSON files.
"""

import json
from pathlib import Path
from typing import List

import pytest
from test_metrics_automated import AutomatedMetricTester, MetricTestSuite


@pytest.fixture
def metric_tester():
    """Fixture providing an AutomatedMetricTester instance."""
    return AutomatedMetricTester()


def load_test_suite_from_file(file_path: str) -> MetricTestSuite:
    """Load a test suite from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return MetricTestSuite(**data)


def get_test_suite_files() -> List[str]:
    """Get all test suite JSON files from the test_cases directory."""
    test_cases_dir = Path(__file__).parent / "test_cases"
    if not test_cases_dir.exists():
        return []

    json_files = list(test_cases_dir.glob("*.json"))
    return [str(f) for f in json_files]


def parametrize_test_suites():
    """Create parametrized test cases for all test suite files."""
    test_files = get_test_suite_files()
    if not test_files:
        pytest.skip("No test suite files found")

    return test_files


class TestAutomatedMetrics:
    """Test class for automated metric testing with pytest."""

    @pytest.mark.parametrize("test_file", parametrize_test_suites())
    def test_metric_suite(self, metric_tester, test_file):
        """Test a complete metric test suite from a JSON file."""
        test_suite = load_test_suite_from_file(test_file)

        # Run all test cases in the suite
        results = metric_tester.run_test_suite(test_suite)

        # Separate failed tests from skipped tests
        failed_tests = [r for r in results if not r["success"] and not r.get("skipped", False)]
        skipped_tests = [r for r in results if r.get("skipped", False)]

        if failed_tests:
            # Create detailed error message
            error_msg = f"Test suite '{test_suite.name}' failed with {len(failed_tests)} failed tests:\n"
            for result in failed_tests:
                error_msg += f"\n  - {result['test_case']}: "
                if result["error"]:
                    error_msg += f"Error: {result['error']}"
                else:
                    error_msg += f"Expected {result['expected']}, got {result['actual']}"

            pytest.fail(error_msg)

        # Log skipped tests
        if skipped_tests:
            print(f"\nSkipped {len(skipped_tests)} tests in '{test_suite.name}':")
            for result in skipped_tests:
                print(f"  - {result['test_case']}: {result.get('skip_reason', 'Unknown reason')}")

        # All non-skipped tests passed
        assert len(failed_tests) == 0, f"Expected all non-skipped tests to pass, but {len(failed_tests)} failed"

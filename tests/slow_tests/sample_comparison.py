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

import math
from dataclasses import asdict
from pathlib import Path

from datasets import Dataset


def _to_plain_list(value):
    """convert a list of tensors to a list of plain values"""
    new_value = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        new_value.append(item)
    return new_value


def _logprobs_approximately_equal(current_logprobs, reference_logprobs):
    """Check if logprobs are sorted in the same order.
    for example:
        current_logprobs = [1.1, 2.1, 3.1]
        reference_logprobs = [1.0, 2.0, 3.0]
        should return True
    """
    if current_logprobs is None and reference_logprobs is None:
        return True
    if current_logprobs is None or reference_logprobs is None:
        return False

    current_logprobs = _to_plain_list(current_logprobs)
    reference_logprobs = _to_plain_list(reference_logprobs)

    # Check if both lists have the same ordering
    # Convert to relative ordering: 0 for smallest, 1 for second smallest, etc.
    current_indices = sorted(range(len(current_logprobs)), key=lambda i: current_logprobs[i])
    reference_indices = sorted(range(len(reference_logprobs)), key=lambda i: reference_logprobs[i])

    return current_indices == reference_indices


def load_sample_details(details_dir: str):
    """Load sample-level details from parquet files in the details directory."""
    details = {}
    details_path = Path(details_dir)

    if not details_path.exists():
        return details

    for parquet_file in details_path.glob("details_*.parquet"):
        # Extract task name from parquet filename, keeping the full task path with "|" separators
        task_name = parquet_file.stem.replace("details_", "").rsplit("_", 1)[
            0
        ]  # Split from right to preserve task name with "|"
        dataset = Dataset.from_parquet(str(parquet_file))
        details[task_name] = list(dataset)

    return details


def _compare_model_responses(current, reference):
    """Compare model response fields between current and reference."""
    sample_diff = {}
    current_resp = current["model_response"]
    reference_resp = reference["model_response"]

    # Get all field names from both responses
    all_fields = set(current_resp.keys()) | set(reference_resp.keys())

    for field_name in all_fields:
        current_val = current_resp.get(field_name)
        reference_val = reference_resp.get(field_name)

        # Special handling for logprobs field
        if field_name in ["input_tokens", "output_tokens"]:
            # input and ouput tokens are lists of tensors, we need to convert
            # them to plain lists
            current_val = _to_plain_list(current_val)
            reference_val = _to_plain_list(reference_val)

            if current_val != reference_val:
                sample_diff["{}_difference".format(field_name)] = {
                    "current": current_val,
                    "reference": reference_val,
                }

    return sample_diff


def _compare_metrics(current, reference):
    """Compare metric fields between current and reference."""
    sample_diff = {}
    current_metrics = current["metric"]
    reference_metrics = reference["metric"]

    metric_diffs = {}
    for metric_name in set(current_metrics.keys()) | set(reference_metrics.keys()):
        current_val = current_metrics.get(metric_name)
        reference_val = reference_metrics.get(metric_name)

        if not math.isclose(current_val, reference_val, abs_tol=0.05):
            metric_diffs[metric_name] = {"current": current_val, "reference": reference_val}

    if metric_diffs:
        sample_diff["metric_differences"] = metric_diffs

    return sample_diff


def _compare_doc_info(current, reference):
    """Compare document information between current and reference."""
    sample_diff = {}
    current_doc = current["doc"]
    reference_doc = reference["doc"]

    if current_doc.get("query") != reference_doc.get("query"):
        sample_diff["query_difference"] = {
            "current": current_doc.get("query"),
            "reference": reference_doc.get("query"),
        }

    if current_doc.get("choices") != reference_doc.get("choices"):
        sample_diff["choices_difference"] = {
            "current": current_doc.get("choices"),
            "reference": reference_doc.get("choices"),
        }

    return sample_diff


def _compare_single_sample(current, reference, sample_index):
    """Compare a single sample between current and reference."""
    sample_diff = {}
    current = asdict(current)

    if "model_response" in current and "model_response" in reference:
        sample_diff.update(_compare_model_responses(current, reference))

    if "metric" in current and "metric" in reference:
        sample_diff.update(_compare_metrics(current, reference))

    if "doc" in current and "doc" in reference:
        sample_diff.update(_compare_doc_info(current, reference))

    if sample_diff:
        sample_diff["sample_index"] = sample_index

    return sample_diff


def compare_sample_details(current_details, reference_details):
    """Compare sample-by-sample details between current and reference results."""
    differences = {}

    for task_name in current_details:
        if task_name not in reference_details:
            differences[task_name] = [{"error": "Task not found in reference results"}]
            continue

        current_samples = current_details[task_name]
        reference_samples = reference_details[task_name]

        if len(current_samples) != len(reference_samples):
            differences[task_name] = [
                {"error": f"Sample count mismatch: current={len(current_samples)}, reference={len(reference_samples)}"}
            ]
            continue

        task_differences = []
        for i, (current, reference) in enumerate(zip(current_samples, reference_samples)):
            sample_diff = _compare_single_sample(current, reference, i)
            if sample_diff:
                task_differences.append(sample_diff)

        if task_differences:
            differences[task_name] = task_differences

    return differences


def _format_single_diff(diff):
    """Format a single sample difference."""
    output = []
    sample_idx = diff.get("sample_index", "unknown")
    output.append("  Sample {}:".format(sample_idx))

    # Handle model response field differences
    for key, value in diff.items():
        if key.endswith("_difference") and key != "metric_differences":
            field_name = key.replace("_difference", "")
            output.append("    {} differs:".format(field_name.title()))
            output.append("      Current: {}".format(value["current"]))
            output.append("      Reference: {}".format(value["reference"]))

    # Handle metric differences
    if "metric_differences" in diff:
        output.append("    Metrics differ:")
        for metric_name, metric_diff in diff["metric_differences"].items():
            output.append(
                "      {}: current={}, reference={}".format(
                    metric_name, metric_diff["current"], metric_diff["reference"]
                )
            )

    return output


def format_sample_differences(differences):
    """Format sample differences into a readable string for test output."""
    if not differences:
        return "No sample-level differences found."

    output = ["Sample-by-sample differences found:"]

    for task_name, task_diffs in differences.items():
        output.append(f"\nTask: {task_name}")

        for diff in task_diffs:
            if "error" in diff:
                output.append(f"  ERROR: {diff['error']}")
                continue

            output.extend(_format_single_diff(diff))

    return "\n".join(output)


def enhance_test_with_sample_comparison(diff, details, reference_details_dir):
    """Enhance test failure messages with sample-by-sample comparison details."""
    # Load reference details if available
    reference_details = load_sample_details(reference_details_dir)

    # Compare sample-by-sample details
    sample_differences = compare_sample_details(details, reference_details)

    # Always check for sample-level differences, even if high-level results match
    if sample_differences:
        sample_diff_message = format_sample_differences(sample_differences)

        if diff != {}:
            # Both high-level and sample-level differences found
            full_message = f"High-level differences found: {diff}\n\n{sample_diff_message}"
        else:
            # Only sample-level differences found (high-level results match)
            full_message = f"High-level results match, but sample-level differences found:\n\n{sample_diff_message}"

        return full_message

    # No sample-level differences found
    if diff != {}:
        # Only high-level differences found
        return f"High-level differences found: {diff}"

    # No differences at any level
    return None

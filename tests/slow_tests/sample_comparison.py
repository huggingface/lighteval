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

from dataclasses import asdict
from pathlib import Path

from datasets import Dataset


def load_sample_details(details_dir: str):
    """Load sample-level details from parquet files in the details directory."""
    details = {}
    details_path = Path(details_dir)

    if not details_path.exists():
        return details

    for parquet_file in details_path.glob("details_*.parquet"):
        task_name = parquet_file.stem.replace("details_", "").split("_")[0]  # Extract task name
        dataset = Dataset.from_parquet(str(parquet_file))
        details[task_name] = list(dataset)

    return details


def _compare_model_responses(current, reference):
    """Compare model response fields between current and reference."""
    sample_diff = {}
    current_resp = current["model_response"]
    reference_resp = reference["model_response"]

    if current_resp.get("text") != reference_resp.get("text"):
        sample_diff["text_difference"] = {
            "current": current_resp.get("text"),
            "reference": reference_resp.get("text"),
        }

    if current_resp.get("logprobs") != reference_resp.get("logprobs"):
        sample_diff["logprobs_difference"] = {
            "current": current_resp.get("logprobs"),
            "reference": reference_resp.get("logprobs"),
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

        if current_val != reference_val:
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
    output.append(f"  Sample {sample_idx}:")

    if "text_difference" in diff:
        output.append("    Text output differs:")
        output.append(f"      Current: {diff['text_difference']['current']}")
        output.append(f"      Reference: {diff['text_difference']['reference']}")

    if "logprobs_difference" in diff:
        output.append("    Logprobs differ:")
        output.append(f"      Current: {diff['logprobs_difference']['current']}")
        output.append(f"      Reference: {diff['logprobs_difference']['reference']}")

    if "metric_differences" in diff:
        output.append("    Metrics differ:")
        for metric_name, metric_diff in diff["metric_differences"].items():
            output.append(
                f"      {metric_name}: current={metric_diff['current']}, reference={metric_diff['reference']}"
            )

    if "query_difference" in diff:
        output.append("    Query differs:")
        output.append(f"      Current: {diff['query_difference']['current']}")
        output.append(f"      Reference: {diff['query_difference']['reference']}")

    if "choices_difference" in diff:
        output.append("    Choices differ:")
        output.append(f"      Current: {diff['choices_difference']['current']}")
        output.append(f"      Reference: {diff['choices_difference']['reference']}")

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

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


from lighteval.metrics.utils.metric_utils import Metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def apply_metric(responses: list[ModelResponse], docs: list[Doc], metrics: list[Metric]):
    # Separate batched and non-batched metrics
    batched_metrics = [m for m in metrics if m.batched_compute]
    non_batched_metrics = [m for m in metrics if not m.batched_compute]

    outputs = []

    # Handle batched metrics first
    batched_outputs = []
    if batched_metrics:
        for metric in batched_metrics:
            metric_outputs = metric.compute_sample(responses=responses, docs=docs)
            batched_outputs.append(metric_outputs)

    # Initialize outputs with the correct structure
    for i in range(len(docs)):
        output = {}

        # Add batched metric results for this sample
        for metric, metric_outputs in zip(batched_metrics, batched_outputs):
            # Batched metrics may return either:
            # 1) list[dict[str, Any]] with one dict per sample, or
            # 2) dict[str, list[Any]] with one list per metric name.
            # Support both shapes for single and grouped metrics.
            if isinstance(metric_outputs, list):
                output.update(metric_outputs[i])
            else:
                metric_names = metric.metric_name if isinstance(metric.metric_name, list) else [metric.metric_name]
                for metric_name in metric_names:
                    output[metric_name] = metric_outputs[metric_name][i]

        # Add non-batched metric results for this sample
        for metric in non_batched_metrics:
            output.update(
                metric.compute_sample(
                    model_response=responses[i],
                    doc=docs[i],
                )
            )

        outputs.append(output)

    return outputs

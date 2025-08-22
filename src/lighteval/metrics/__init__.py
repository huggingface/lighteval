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


from lighteval.metrics.metrics import Metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def apply_metric(responses: list[ModelResponse], docs: list[Doc], metrics: list[Metric]):
    outputs = []
    for metric in metrics:
        if metric.batched_compute:
            outputs_per_metrics: list = []

            outputs_per_metrics.append(metric.compute(responses=responses, docs=docs))

            # We merge the outputs per metric in a list of dict for each sample
            # example: [{metric1_sample1, metric2_sample1}, {metric1_sample2, metric2_sample2}]
            for i in range(len(docs)):
                output = {}
                for metric_outputs in outputs_per_metrics:
                    output.update(metric_outputs[i])
                outputs.append(output)

        else:
            for model_response, doc in zip(responses, docs):
                output = {}
                for metric in metrics:
                    output.update(
                        metric.compute(
                            model_response=model_response,
                            doc=doc,
                        )
                    )
                outputs.append(output)

    return outputs

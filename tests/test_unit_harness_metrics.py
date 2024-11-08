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

import pytest

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


PATH_TO_HARNESS_METRICS = os.path.join(os.path.dirname(__file__), "reference_scores/harness_metrics.json")

POSSIBLE_METRICS = Metrics.all_metrics()


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Initializes the main test setup. This function is automatically called by pytest and
    should not be called manually.

    Every function with "model_input" as arguments will be sent the "parameters".
    This function will be run only once, ensuring that each model is run only once on the selected tasks.
    (This is better than using fixtures as fixtures are re-run once for each test, which is not a behavior we want).
    """
    parameters = []

    # If model_input is a test function argument
    # (= the function requires a fixture)
    if "prompt_inputs" in metafunc.fixturenames:
        with open(PATH_TO_HARNESS_METRICS) as f:
            metric_to_examples = json.load(f)

            for metric, examples in metric_to_examples.items():
                for task_name, examples_list in examples.items():
                    parameters.append((metric, task_name, examples_list))
        metafunc.parametrize("prompt_inputs", parameters, scope="session")


def test_model_prediction(prompt_inputs: tuple[str, str, list]):
    """Evaluates a model on a full task - is parametrized using pytest_generate_test"""
    metric, task_name, examples = prompt_inputs
    metric_name = metric
    metric = Metrics[metric].value
    print(metric_name, task_name)
    for example in examples:
        formatted_doc = {
            k: v
            for k, v in example.items()
            if k in ["full_prompt", "choices", "gold_index", "original_query", "specific"]
        }
        print(formatted_doc)
        formatted_doc["query"] = formatted_doc.pop("full_prompt")
        formatted_doc = Doc(**formatted_doc)
        error_msg = f"Metric {metric_name} failed on input {formatted_doc} from task {task_name}.\n"

        results = [ModelResponse(result=i, input_tokens=[], generated_tokens=[]) for i in example["predictions"]]
        # todo: update to create list of ModelResults in results
        metric_result = apply_metric(
            sample_ids=["0"], metric=metric, responses=[results], formatted_docs=[formatted_doc]
        )[0]
        assert metric_result is not None, error_msg
        metric_result = {k: list(v) if isinstance(v, tuple) else v for k, v in metric_result.items()}

        metric_reference = {k: v for k, v in example.items() if k in POSSIBLE_METRICS}
        error_msg += f"Prediction: {results}\n"
        error_msg += f"Reference: {metric_reference}\n"
        error_msg += f"Returned : {metric_result}"

        for key in metric_result.keys():
            if type(metric_result[key]) in [
                LogprobCorpusMetricInput,
                GenerativeCorpusMetricInput,
                PerplexityCorpusMetricInput,
            ]:
                cur_result_list = as_list(metric_result[key].to_dict())
            else:
                cur_result_list = as_list(metric_result[key])
            cur_ref_list = as_list(metric_reference[key])

            # item wise comparison of lists
            if isinstance(cur_result_list[0], list):
                for res, ref in zip(cur_result_list, cur_ref_list):
                    try:
                        assert res == pytest.approx(ref, rel=1e-8), error_msg
                    except Exception as e:
                        assert False, error_msg + "\n" + str(e)
            else:
                try:
                    assert cur_result_list == pytest.approx(cur_ref_list, rel=1e-8), error_msg
                except Exception as e:
                    assert False, error_msg + "\n" + str(e)


def apply_metric(sample_ids, metric, responses, formatted_docs: list[Doc]):
    method = LightevalTask._get_metric_method_from_category(metric.category)
    cur_outputs = method(sample_ids=sample_ids, metrics=[metric], responses=responses, formatted_docs=formatted_docs)
    return cur_outputs

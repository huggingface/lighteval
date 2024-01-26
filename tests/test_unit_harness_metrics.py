import json
import os

import pytest

from lighteval.metrics import (
    apply_generative_logprob_metric,
    apply_generative_metric,
    apply_multichoice_metric,
    apply_multichoice_metric_one_token,
    apply_perplexity_metric,
    apply_target_perplexity_metric,
)
from lighteval.metrics.metrics import MetricCategory, Metrics
from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from lighteval.models.model_output import ModelReturn
from lighteval.tasks.requests import Doc
from lighteval.utils import as_list


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
    print(metric, task_name)
    for example in examples:
        formatted_doc = {
            k: v
            for k, v in example.items()
            if k in ["full_prompt", "choices", "gold_index", "original_query", "specific"]
        }
        print(formatted_doc)
        formatted_doc["query"] = formatted_doc.pop("full_prompt")
        formatted_doc = Doc(**formatted_doc)
        error_msg = f"Metric {metric} failed on input {formatted_doc} from task {task_name}.\n"

        results = [ModelReturn(result=i, input_tokens=[], generated_tokens=[]) for i in example["predictions"]]
        # todo: update to create list of ModelResults in results
        metric_result = apply_metric(metric=metric, results=results, formatted_doc=formatted_doc)
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

            # itemwise comparision of lists
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


def apply_metric(metric, results, formatted_doc: Doc):
    if Metrics[metric].value.category == MetricCategory.TARGET_PERPLEXITY:
        _, cur_outputs = apply_target_perplexity_metric(results=results, formatted_doc=formatted_doc, metrics=[metric])
        return cur_outputs
    if Metrics[metric].value.category == MetricCategory.PERPLEXITY:
        _, cur_outputs = apply_perplexity_metric(results=results, formatted_doc=formatted_doc, metrics=[metric])
        return cur_outputs
    if Metrics[metric].value.category == MetricCategory.GENERATIVE:
        _, cur_outputs = apply_generative_metric(results=results, formatted_doc=formatted_doc, metrics=[metric])
        return cur_outputs
    if Metrics[metric].value.category == MetricCategory.GENERATIVE_LOGPROB:
        _, cur_outputs = apply_generative_logprob_metric(
            results=results, formatted_doc=formatted_doc, metrics=[metric]
        )
        return cur_outputs
    if Metrics[metric].value.category == MetricCategory.MULTICHOICE:
        _, cur_outputs = apply_multichoice_metric(results=results, formatted_doc=formatted_doc, metrics=[metric])
        return cur_outputs
    if Metrics[metric].value.category == MetricCategory.MULTICHOICE_ONE_TOKEN:
        _, cur_outputs = apply_multichoice_metric_one_token(
            results=results, formatted_doc=formatted_doc, metrics=[metric]
        )
        return cur_outputs
    else:
        raise Exception(f"Metric {metric} not found.")

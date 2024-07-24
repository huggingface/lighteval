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

import re

from lighteval.metrics.metrics import Metric, MetricCategory
from lighteval.models.model_output import ModelReturn
from lighteval.tasks.requests import Doc
from lighteval.utils import as_list


def apply_target_perplexity_metric(results: list[ModelReturn], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    reference_text = formatted_doc.get_golds()[0]
    current_result = results.pop(0)
    target_logprob = current_result.result[0]
    target_acc = current_result.result[1]

    for metric in metrics:
        if metric.category == MetricCategory.TARGET_PERPLEXITY:
            outputs.update(
                metric.compute(logprobs=target_logprob, target_acc=target_acc, reference_text=reference_text)
            )

    return results, outputs


def apply_perplexity_metric(results: list[ModelReturn], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    current_result = results.pop(0)
    # Sometimes, processing was added for the log processings
    # that we don't want to include when computing the sentence length
    # Check if we want to keep this or not
    if formatted_doc.original_query not in [None, ""]:
        reference_text = formatted_doc.original_query
    else:
        reference_text = formatted_doc.query

    for metric in metrics:
        if metric.category == MetricCategory.PERPLEXITY:
            outputs.update(metric.compute(logprobs=current_result.result, reference_text=reference_text))

    return results, outputs


def apply_generative_metric(
    results: list[ModelReturn], formatted_doc: Doc, metrics: list[Metric], output_regex=None, max_num_samples=1
):
    outputs = {}

    # Post processing prediction
    preds_raw = as_list(results.pop(0).result)
    preds = []

    for pred_raw in preds_raw:
        if output_regex is not None:
            pred = next(iter(re.findall(output_regex, pred_raw)), "")
        else:
            pred = pred_raw
        preds.append(pred)

    # Extracting gold
    try:
        golds = formatted_doc.get_golds()
    except (KeyError, IndexError):
        golds = None

    # Specific process for HELM like evals # hrm
    # if "label_to_choices" in formatted_doc:
    if formatted_doc.specific is not None and "label_to_choices" in formatted_doc.specific:
        # Helm predicts on labels keys (A/B/C/D), but computes metrics on choices
        preds = [formatted_doc.specific["label_to_choices"].get(p) for p in preds]
        golds = [formatted_doc.specific["label_to_choices"][g] for g in golds]

    for metric in metrics:
        if metric.category == MetricCategory.GENERATIVE:
            outputs.update(
                metric.compute(
                    golds=golds,
                    predictions=as_list(preds[0]) if max_num_samples > 1 else preds,
                    formatted_doc=formatted_doc,
                )
            )
        if metric.category == MetricCategory.GENERATIVE_LOGPROB:
            outputs.update(
                metric.compute(
                    golds=golds,
                    predictions=as_list(preds[0]) if max_num_samples > 1 else preds,
                    formatted_doc=formatted_doc,
                )
            )
        if metric.category == MetricCategory.GENERATIVE_SAMPLING:
            outputs.update(metric.compute(golds=golds, predictions=preds, formatted_doc=formatted_doc))

    return results, outputs


def apply_multichoice_metric(results: list[ModelReturn], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    mc_results = results[: len(formatted_doc.choices)]
    if len(formatted_doc.choices) <= 1:
        raise ValueError(
            "You can't use a multi choice metric with only one choice. Use `acc_golds_likelihood` instead."
        )

    # Todo: make better system with return_bool_score instead of taking first element
    choices_logprob = [mc_results[i].result[0] for i in range(len(formatted_doc.choices))]  # sum(
    gold_ixs = as_list(formatted_doc.gold_index)

    for metric in metrics:
        if metric.category == MetricCategory.MULTICHOICE:
            outputs.update(
                metric.compute(choices_logprob=choices_logprob, gold_ixs=gold_ixs, formatted_doc=formatted_doc)
            )
    return results[len(formatted_doc.choices) :], outputs


def apply_multichoice_metric_one_token(results: list[ModelReturn], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    choices_logprob = results.pop(0).result
    gold_ixs = as_list(formatted_doc.gold_index)

    for metric in metrics:
        if metric.category == MetricCategory.MULTICHOICE_ONE_TOKEN:
            outputs.update(
                metric.compute(choices_logprob=choices_logprob, gold_ixs=gold_ixs, formatted_doc=formatted_doc)
            )

    return results, outputs


def apply_llm_as_judge_metric(results: list[ModelReturn], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    predictions = results.pop(0).result

    for metric in metrics:
        if metric.category in [MetricCategory.LLM_AS_JUDGE_MULTI_TURN, MetricCategory.LLM_AS_JUDGE]:
            outputs.update(metric.compute(predictions=predictions, formatted_doc=formatted_doc))

    return results, outputs

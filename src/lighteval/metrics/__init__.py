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
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


def apply_target_perplexity_metric(results: list[ModelResponse], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}

    target_golds = formatted_doc.get_golds()
    assert len(results) == len(target_golds), "You should return as many results as there are golds"
    target_logprobs = [res.result[0] for res in results]
    argmax_logits_eq_gold_list = [res.result[1] for res in results]
    target_tokens = [res.generated_tokens for res in results]

    for metric in metrics:
        if metric.category == MetricCategory.TARGET_PERPLEXITY:
            outputs.update(
                metric.compute(
                    logprobs=target_logprobs,
                    argmax_logits_eq_gold_list=argmax_logits_eq_gold_list,
                    reference_texts=target_golds,
                    target_tokens=target_tokens,
                )
            )

    return outputs


def apply_perplexity_metric(results: list[ModelResponse], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    if len(results) > 1:
        raise Exception("You returned more than one result for a sample with a perplexity metric.")
    results = results[0]

    # Sometimes, processing was added for the log processings
    # that we don't want to include when computing the sentence length
    # Check if we want to keep this or not
    if formatted_doc.original_query not in [None, ""]:
        reference_text = formatted_doc.original_query
    else:
        reference_text = formatted_doc.query

    for metric in metrics:
        if metric.category == MetricCategory.PERPLEXITY:
            outputs.update(metric.compute(logprobs=[results.result], reference_texts=[reference_text]))

    return outputs


def apply_generative_metric(
    results: list[ModelResponse], formatted_doc: Doc, metrics: list[Metric], output_regex=None, max_num_samples=1
):
    outputs = {}

    if len(results) > 1:
        raise Exception("You returned more than one result for a sample with a generative metric.")
    results = results[0]

    # Post processing prediction
    preds_raw = as_list(results.result)
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

    return outputs


def apply_multichoice_metric(results: list[ModelResponse], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    n_choices = len(formatted_doc.choices)
    is_pmi_category = all(metric.category == MetricCategory.MULTICHOICE_PMI for metric in metrics)

    if n_choices <= 1:
        raise ValueError(
            "You can't use a multi choice metric with only one choice. Use `acc_golds_likelihood` instead."
        )

    if not is_pmi_category and len(results) != len(formatted_doc.choices):
        raise Exception(
            f"You shoud have returned as many model outputs as choices when using an multi choice metric. Returned {len(results)} instead of {len(formatted_doc.choices)}"
        )

    if is_pmi_category and len(results) != n_choices * 2:
        raise Exception(
            f"You shoud have returned twice as many model outputs as choices when using an probability multi choice metric. Returned {len(results)} instead of {n_choices * 2} (conditioned and unconditioned)"
        )

    mc_results = results[:n_choices]
    # Todo: make better system with return_bool_score instead of taking first element
    conditioned_lp = [res.result[0] for res in mc_results]
    unconditioned_lp = None
    if is_pmi_category:
        unconditioned_lp = [res.result[0] for res in results[n_choices : n_choices * 2]]

    gold_ixs = as_list(formatted_doc.gold_index)
    choices_tokens = [res.generated_tokens for res in mc_results]

    for metric in metrics:
        if metric.category == MetricCategory.MULTICHOICE_PMI or metric.category == MetricCategory.MULTICHOICE:
            outputs.update(
                metric.compute(
                    gold_ixs=gold_ixs,
                    choices_logprob=conditioned_lp,
                    unconditioned_logprob=unconditioned_lp,
                    choices_tokens=choices_tokens,
                    formatted_doc=formatted_doc,
                )
            )
    return outputs


def apply_multichoice_metric_one_token(results: list[ModelResponse], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    if len(results) > 1:
        raise Exception("You returned more than one result for a sample with a gmultichoice metric on only one token.")
    results = results[0]
    choices_logprob = results.result
    choices_texts = formatted_doc.choices
    gold_ixs = as_list(formatted_doc.gold_index)

    for metric in metrics:
        if metric.category == MetricCategory.MULTICHOICE_ONE_TOKEN:
            outputs.update(
                metric.compute(
                    choices_logprob=choices_logprob,
                    # Neither token or PMI are supported for this metric
                    unconditioned_logprob=None,
                    choices_tokens=None,
                    choices_texts=choices_texts,
                    gold_ixs=gold_ixs,
                    formatted_doc=formatted_doc,
                )
            )

    return outputs


def apply_llm_as_judge_metric(results: list[ModelResponse], formatted_doc: Doc, metrics: list[Metric]):
    outputs = {}
    if len(results) > 1:
        raise Exception("You returned more than one result for a sample with an llm as a judge metric.")
    results = results[0]

    predictions = results.result

    for metric in metrics:
        if metric.category in [MetricCategory.LLM_AS_JUDGE_MULTI_TURN, MetricCategory.LLM_AS_JUDGE]:
            outputs.update(metric.compute(predictions=predictions, formatted_doc=formatted_doc))

    return outputs

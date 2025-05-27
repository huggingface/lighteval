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


from lighteval.metrics.metrics import Metric, MetricCategory
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


def apply_target_perplexity_metric(responses: list[ModelResponse], docs: list[Doc], metrics: list[Metric]):
    outputs = []

    for model_reponse, formatted_doc in zip(responses, docs):
        target_golds = formatted_doc.get_golds()
        output = {}

        assert len(model_reponse.logprobs) == len(target_golds), "You should return as many results as there are golds"

        for metric in metrics:
            if metric.category == MetricCategory.TARGET_PERPLEXITY:
                output.update(
                    metric.compute(
                        logprobs=model_reponse.logprobs,
                        argmax_logits_eq_gold_list=model_reponse.argmax_logits_eq_gold,
                        reference_texts=target_golds,
                        target_tokens=model_reponse.output_tokens,
                    )
                )
        outputs.append(output)

    return outputs


def apply_perplexity_metric(responses: list[ModelResponse], docs: list[Doc], metrics: list[Metric]):
    outputs = []
    for model_reponse, formatted_doc in zip(responses, docs):
        output = {}
        if len(model_reponse.logprobs) > 1:
            raise Exception("You returned more than one result for a sample with a perplexity metric.")

        # Sometimes, processing was added for the log processings
        # that we don't want to include when computing the sentence length
        # Check if we want to keep this or not
        if formatted_doc.original_query not in [None, ""]:
            reference_text = formatted_doc.original_query
        else:
            reference_text = formatted_doc.query

        for metric in metrics:
            if metric.category == MetricCategory.PERPLEXITY:
                output.update(metric.compute(logprobs=model_reponse.logprobs, reference_texts=[reference_text]))

        outputs.append(output)

    return outputs


def apply_generative_metric(  # noqa: C901
    responses: list[ModelResponse],
    docs: list[Doc],
    metrics: list[Metric],
):
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
            for model_response, formatted_doc in zip(responses, docs):
                output = {}

                try:
                    golds = formatted_doc.get_golds()
                except (KeyError, IndexError):
                    golds = None

                for metric in metrics:
                    output.update(
                        metric.compute(
                            golds=golds,
                            predictions=model_response.text,
                            docs=formatted_doc,
                        )
                    )
                outputs.append(output)

    return outputs


def apply_multichoice_metric(responses: list[ModelResponse], docs: list[Doc], metrics: list[Metric]):
    outputs = []
    for model_reponse, formatted_doc in zip(responses, docs):
        output = {}
        n_choices = len(formatted_doc.choices)
        is_pmi_category = all(metric.category == MetricCategory.MULTICHOICE_PMI for metric in metrics)

        if n_choices <= 1:
            raise ValueError(
                "You can't use a multi choice metric with only one choice. Use `acc_golds_likelihood` instead."
            )

        if not is_pmi_category and len(model_reponse.logprobs) != len(formatted_doc.choices):
            raise Exception(
                f"You shoud have returned as many model outputs as choices when using an multi choice metric. Returned {len(model_reponse.logprobs)} instead of {len(formatted_doc.choices)}"
            )

        if is_pmi_category and len(model_reponse.logprobs) != n_choices * 2:
            raise Exception(
                f"You shoud have returned twice as many model outputs as choices when using an probability multi choice metric. Returned {len(model_reponse.logprobs)} instead of {n_choices * 2} (conditioned and unconditioned)"
            )

        conditioned_logprobs = model_reponse.logprobs[:n_choices]
        # Todo: make better system with return_bool_score instead of taking first element
        unconditioned_logprobs = None
        if is_pmi_category:
            unconditioned_logprobs = model_reponse.logprobs[n_choices : n_choices * 2]

        gold_ixs = as_list(formatted_doc.gold_index)
        choices_tokens = model_reponse.output_tokens[:n_choices]

        for metric in metrics:
            if metric.category == MetricCategory.MULTICHOICE_PMI or metric.category == MetricCategory.MULTICHOICE:
                output.update(
                    metric.compute(
                        gold_ixs=gold_ixs,
                        choices_logprob=conditioned_logprobs,
                        unconditioned_logprob=unconditioned_logprobs,
                        choices_tokens=choices_tokens,
                        formatted_doc=formatted_doc,
                    )
                )
        outputs.append(output)

    return outputs


def apply_llm_as_judge_metric(responses: list[ModelResponse], docs: list[Doc], metrics: list[Metric]):
    """
    Apply the LLM as judge metric to the responses. The batching is managed at the judge level.
    """
    # outputs per metric is a list containing a list of dict for each metric
    # example: [[{metric1_sample1}, {metric1_sample2}], [{metric2_sample1}, {metric2_sample2}]]
    outputs_per_metrics: list[list[dict]] = []

    for metric in metrics:
        if metric.category in [MetricCategory.LLM_AS_JUDGE_MULTI_TURN, MetricCategory.LLM_AS_JUDGE]:
            outputs_per_metrics.append(metric.compute(responses=responses, doc=docs))

    # We merge the outputs per metric in a list of dict for each sample
    # example: [{metric1_sample1, metric2_sample1}, {metric1_sample2, metric2_sample2}]
    outputs = []
    for i in range(len(docs)):
        output = {}
        for metric_outputs in outputs_per_metrics:
            output.update(metric_outputs[i])
        outputs.append(output)

    return outputs

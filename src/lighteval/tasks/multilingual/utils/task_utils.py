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


from typing import Literal

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric, multilingual_extractive_match_metric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import Metric
from lighteval.tasks.templates.utils.formulation import Formulation, MCFFormulation
from lighteval.utils.language import Language


EvalType = Literal["generative", "logprobs"]


def normalize_subset(subset: str) -> str:
    return subset.replace(" ", "_").replace("(", "").replace(")", "").lower()


def get_metrics_for_mcq_formulation(
    formulation: Formulation, language: Language, metrics: list[Metric], eval_type: EvalType
) -> list[Metric]:
    """
    Choose the appropriate metrics for the given formulation otherwise fallback to the original metrics.
    """
    match (formulation, eval_type):
        # In case of
        case MCFFormulation(choice_prefix=("Letters" | "Numbers")), "logprobs":
            return [loglikelihood_acc_metric(normalization=None)]
        case _, "logprobs":
            return metrics
        case MCFFormulation(cot=False), "generative":
            return [Metrics.exact_match, Metrics.prefix_exact_match]
        case MCFFormulation(cot=True), "generative":
            return [
                multilingual_extractive_match_metric(language, target_for_extraction=formulation.choice_prefix),
                Metrics.prefix_exact_match,
            ]
        case _:
            return metrics


def get_cot_generaion_size(cot: bool, generation_size: int) -> int:
    if cot:
        return -1
    return generation_size


def get_cot_stop_sequence(cot: bool, stop_sequence: list[str]) -> list[str] | None:
    if cot:
        return None
    return stop_sequence

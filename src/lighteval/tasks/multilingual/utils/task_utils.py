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
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.metrics.utils.metric_utils import Metric
from lighteval.tasks.templates.utils.formulation import CFFormulation, Formulation, HybridFormulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


def normalize_subset(subset: str) -> str:
    return subset.replace(" ", "_").replace("(", "").replace(")", "").lower()

def get_metrics_for_mcq(
    formulation: Formulation, language: Language, logprobs_metrics: list[Metric]
) -> list[Metric]:
    """
    Choose the appropriate metrics given formulation and eval type
    """
    match (formulation):
        # We run in generative manner for COT tasks
        case MCFFormulation(cot=True):
            return [
                multilingual_extractive_match_metric(
                    language,
                    gold_extraction_target=(IndicesExtractionConfig(prefix_for_extraction=formulation.choice_prefix),),
                ),
            ]

        case MCFFormulation(choice_prefix=("Letters" | "Numbers"), cot=False):
            return [loglikelihood_acc_metric(normalization=LogProbTokenNorm())]
        case CFFormulation() | HybridFormulation():
            return logprobs_metrics
        case _:
            raise ValueError(f"Combination of formulation {formulation=} not supported")


def get_cot_generaion_size(cot: bool, generation_size: int) -> int | None:
    return 1024


def get_cot_stop_sequence(language: Language, formulation: Formulation) -> list[str] | None:
    stop_sequence = ["\n"] if not formulation.cot else []
    try:
        trans = TRANSLATION_LITERALS[language]
        # Ensure it's on a new line as otherwise LLM's sometimes like to generate:
        # "**Répondez à la question" or "1. **Comprendre la question" in their cot generations
        return [
            f"\n{trans.question_word}{trans.colon}",
            f"\n{trans.question_word.capitalize()}{trans.colon}",
        ] + stop_sequence
    except (AttributeError, KeyError):
        return stop_sequence

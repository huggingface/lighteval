from typing import Callable

import numpy as np
from aenum import extend_enum
from transformers import GPT2Tokenizer

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import F1_score, SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod


tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")


class TTC_score(SampleLevelComputation):
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
        strip_strings: bool = False,
    ):
        if aggregation_function is None:
            aggregation_function = np.mean

        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        response = model_response.final_text[0]
        all_tokens = tokenizer.encode(response)

        for i in range(1, len(all_tokens) + 1):
            partial_response = tokenizer.decode(all_tokens[:i])
            f1 = F1_score()
            partial_model_response = ModelResponse(
                text=[partial_response],
                output_tokens=[],
                logprobs=[],
            )
            score = f1.compute(doc, partial_model_response)
            if score >= 0.9:
                return i / len(all_tokens)

        return 1.0


my_custom_ttc_metric = SampleLevelMetric(
    metric_name="ttc",
    higher_is_better=False,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=TTC_score(),
    corpus_level_fn=np.mean,
)

extend_enum(Metrics, "CUSTOM_TTC", my_custom_ttc_metric)

if __name__ == "__main__":
    print("Imported metric")

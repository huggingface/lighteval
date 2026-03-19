from typing import Callable

import numpy as np
from aenum import extend_enum
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import ExactMatches, F1_score, SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod


tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
f1 = F1_score()
em = ExactMatches(type_exact_match="prefix", strip_strings=True)
progress_bar = None


class TTC_score_F1(SampleLevelComputation):
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
        global progress_bar

        if progress_bar is None:
            progress_bar = tqdm(desc="Computing TTC metrics", unit="samples")

        progress_bar.update(1)

        response = model_response.final_text[0]
        all_tokens = tokenizer.encode(response)
        partial_model_response = ModelResponse(
            text=[""],
            input_tokens=model_response.input_tokens,
            output_tokens=[],
            logprobs=[],
        )

        for i in range(1, len(all_tokens) + 1):
            partial_response = tokenizer.decode(all_tokens[:i])
            partial_model_response.text = [partial_response]
            partial_model_response.output_tokens = all_tokens[:i]
            score = f1.compute(doc, partial_model_response)
            if score >= 0.9:
                return i / len(all_tokens)

        return 1.0


class TTC_score_EM(SampleLevelComputation):
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

        partial_model_response = ModelResponse(
            text=[""],
            input_tokens=model_response.input_tokens,
            output_tokens=[],
            logprobs=[],
        )

        for i in range(1, len(all_tokens) + 1):
            partial_response = tokenizer.decode(all_tokens[:i])
            partial_model_response.text = [partial_response]
            partial_model_response.output_tokens = all_tokens[:i]
            score = em.compute(doc, partial_model_response)
            if score >= 1.0:
                return i / len(all_tokens)

        return 1.0


my_custom_ttc_f1_metric = SampleLevelMetric(
    metric_name="ttc_f1",
    higher_is_better=False,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=TTC_score_F1(),
    corpus_level_fn=np.mean,
)

my_custom_ttc_em_metric = SampleLevelMetric(
    metric_name="ttc_em",
    higher_is_better=False,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=TTC_score_EM(),
    corpus_level_fn=np.mean,
)

extend_enum(Metrics, "CUSTOM_TTC_F1", my_custom_ttc_f1_metric)
extend_enum(Metrics, "CUSTOM_TTC_EM", my_custom_ttc_em_metric)

if __name__ == "__main__":
    print("Imported metric")

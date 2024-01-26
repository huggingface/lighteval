from dataclasses import dataclass, field
from typing import Optional, Union

import torch


@dataclass
class ModelReturn:  # @clefourrier: could probably an abstract class, but it might make the code too complex
    result: Union[tuple, list, str]
    input_tokens: list[int] = field(default_factory=list)  # model inputs
    generated_tokens: list[int] = field(default_factory=list)  # model generations
    truncated_tokens_count: Optional[int] = None  # How many tokens truncated
    padded_tokens_count: Optional[int] = None  # How many tokens of padding

    def get_result_for_eval(self):
        raise NotImplementedError()


@dataclass
class LoglikelihoodReturn(ModelReturn):
    # Float: Total log prob of the continuation
    # Optional(Bool): Whether the continuation is greedy (= all the tokens in the continuation are argmax of prob)
    result: Union[tuple[float, bool], float] = field(default_factory=tuple[float, bool])

    def get_result_for_eval(self):
        return self.result


@dataclass
class LoglikelihoodSingleTokenReturn(ModelReturn):
    # Log probs of the various single token options
    result: list[float] = field(default_factory=list)

    def get_result_for_eval(self):
        return self.result


@dataclass
class GenerateReturn(ModelReturn):
    result: str = field(default_factory=str)  # generated text continuation
    logits: Optional[list[float]] = None  # Generated text logits

    def get_result_for_eval(self):
        return self.result if self.logits is None else (self.result, self.logits)


@dataclass
class Batch:
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    input_lengths: list[int]
    truncated: list[int]
    padded: list[int]

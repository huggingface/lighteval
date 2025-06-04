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
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Union

from lighteval.utils.utils import as_list


class SamplingMethod(str, Enum):
    """
    Enum representing different sampling methods for text generation.
    """

    GENERATIVE = "GENERATIVE"
    LOGPROBS = "LOGPROBS"


@dataclass(slots=True)
class Doc:
    """
    Dataclass used to represent the content of a task example
    almost every field is optional, but some tasks require some fields to be present.
    When adding a new task, please add the required fields to the doc class.
    Each task will have a different set of fields needed.
    """

    query: str
    choices: list[str]
    gold_index: Union[int, list[int]]
    id: str = ""
    original_query: Optional[str] = ""  # the query before preprocessing, if stored
    specific: dict | None = None  # Information which is specific to the current eval
    task_name: str = ""
    system_prompt: str | None = None  # system prompt to use for the model, if any
    full_prompt: Optional[str] = None  # full prompt to use for the model, if any

    # For few-shot
    instruction: Optional[str] = ""
    fewshot_sorting_class: Optional[str] = None  # class to use to select balanced few-shot samples

    # Filled when parsing and adding the few-shot context
    num_asked_few_shots: int = -1
    num_effective_few_shots: int = -1

    # Uncoditioned query is used for PMI normalization, that's
    # log P(choice | Query) - log P(choice | Unconditioned Query)
    # The uncoditioned query shouldn't contain any information about the task, thus usually it's empty string or 'Answer:'.
    unconditioned_query: Optional[str] = None

    fewshot_samples: list = field(default_factory=list)
    sampling_methods: list[SamplingMethod] = field(default_factory=list)

    # Generation parameters
    generation_size: int | None = None  # number of tokens to generate for each sample
    do_sample: bool = False  # whether to sample or not
    stop_sequence: list[str] = field(default_factory=list)
    use_logits: bool = False  # whether to use logits for the generation or not
    num_samples: int = 1  # number of samples to generate for each sample

    def __post_init__(self):
        if self.instruction is None:
            self.instruction = ""

    def get_golds(self):
        """Return gold targets extracted from the target dict"""
        gold_indices = as_list(self.gold_index)
        golds = []
        for gold_ix in gold_indices:
            golds.extend(as_list(self.choices[gold_ix]))
        return golds

    def __repr__(self):
        doc_dict = asdict(self)
        return json.dumps(doc_dict)

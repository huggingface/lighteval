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
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

from huggingface_hub import TextGenerationInputGrammarType

from lighteval.utils.utils import as_list


if TYPE_CHECKING:
    from PIL.Image import Image


class RequestType(Enum):
    LOGLIKELIHOOD = auto()
    LOGLIKELIHOOD_SINGLE_TOKEN = auto()
    LOGLIKELIHOOD_ROLLING = auto()
    GREEDY_UNTIL = auto()
    GREEDY_UNTIL_MULTI_TURN = auto()


@dataclass
class Request:
    """
    Represents a request for a specific task, example and request within that
    example in the evaluation process.
    For example in the task "boolq", the example "Is the sun hot?" and the
    requests for that example "Is the sun hot? Yes" and "Is the sun hot? No".

    Attributes:
        task_name (str): The name of the task.
        sample_index (int): The index of the example.
        request_index (int): The index of the request.
        context (str): The context for the request.
        metric_categories (list[MetricCategory]): All the metric categories which concern this request
    """

    task_name: str
    sample_index: int
    request_index: int
    context: str
    metric_categories: list["MetricCategory"]  # noqa F821


@dataclass
class LoglikelihoodRequest(Request):
    """
    Represents a request for log-likelihood evaluation.

    Attributes:
        choice (str): The choice to evaluate the log-likelihood for.
        request_type (RequestType): The type of the request (LOGLIKELIHOOD).
    """

    choice: str
    request_type = RequestType.LOGLIKELIHOOD
    tokenized_context: list[int] = None
    tokenized_continuation: list[int] = None
    images: Optional[list["Image"]] = None


@dataclass
class LoglikelihoodSingleTokenRequest(Request):
    """
    Represents a request for calculating the log-likelihood of a single token.
    Faster because we can get all the loglikelihoods in one pass.

    Attributes:
        choices (list[str]): The list of token choices.
        request_type (RequestType): The type of the request.
    """

    choices: list[str]
    request_type = RequestType.LOGLIKELIHOOD_SINGLE_TOKEN
    tokenized_context: list[int] = None
    tokenized_continuation: list[int] = None
    images: Optional[list["Image"]] = None


@dataclass
class LoglikelihoodRollingRequest(Request):
    """
    Represents a request for log-likelihood rolling evaluation.

    Inherits from the base Request class.
    """

    request_type = RequestType.LOGLIKELIHOOD_ROLLING
    tokenized_context: list[int] = None
    tokenized_continuation: list[int] = None
    images: Optional[list["Image"]] = None


@dataclass
class GreedyUntilRequest(Request):
    """
    Represents a request for generating text using the Greedy-Until algorithm.

    Attributes:
        stop_sequence (str): The sequence of tokens that indicates when to stop generating text.
        generation_size (int): The maximum number of tokens to generate.
        generation_grammar (TextGenerationInputGrammarType): The grammar to generate completion according to.
            Currently only available for TGI models.
        request_type (RequestType): The type of the request, set to RequestType.GREEDY_UNTIL.
    """

    stop_sequence: Union[str, tuple[str], list[str]]
    generation_size: Union[int, None]
    generation_grammar: Union[TextGenerationInputGrammarType, None] = None
    request_type = RequestType.GREEDY_UNTIL
    tokenized_context: list[int] = None
    num_samples: int = None
    do_sample: bool = False
    use_logits: bool = False
    images: Optional[list["Image"]] = None


@dataclass
class GreedyUntilMultiTurnRequest(Request):
    """
    Represents a request for generating text using the Greedy-Until algorithm.

    Attributes:
        stop_sequence (str): The sequence of tokens that indicates when to stop generating text.
        generation_size (int): The maximum number of tokens to generate.
        request_type (RequestType): The type of the request, set to RequestType.GREEDY_UNTIL.
    """

    stop_sequence: str
    generation_size: int
    request_type = RequestType.GREEDY_UNTIL_MULTI_TURN
    use_logits: bool = False
    images: Optional[list["Image"]] = None


class SampleUid(NamedTuple):
    """
    Represents the identifier for an example in a task.

    Attributes:
        task_name (str): The name of the task in `name|num_fewshot` format.
        doc_id_seed (str): The document id with the seed used for few_shot appended at the end.
    """

    task_name: str
    doc_id_seed: str


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
    original_query: Optional[str] = ""  # the query before preprocessing, if stored
    specific: dict = None  # Information which is specific to the current eval
    task_name: str = ""

    # For few-shot
    instruction: Optional[str] = ""
    fewshot_sorting_class: Optional[str] = None  # class to use to select balanced few-shot samples

    # Filled when parsing and adding the few-shot context
    ctx: Optional[str] = ""
    num_asked_few_shots: int = -1
    num_effective_few_shots: int = -1

    # Uncoditioned query is used for PMI normalization, that's
    # log P(choice | Query) - log P(choice | Unconditioned Query)
    # The uncoditioned query shouldn't contain any information about the task, thus usually it's empty string or 'Answer:'.
    unconditioned_query: Optional[str] = None

    # For multi-modal tasks
    images: Optional[list["Image"]] = None

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

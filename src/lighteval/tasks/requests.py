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
from typing import NamedTuple, Optional, Union

from lighteval.utils import as_list


class RequestType(Enum):
    LOGLIKELIHOOD = auto()
    LOGLIKELIHOOD_SINGLE_TOKEN = auto()
    LOGLIKELIHOOD_ROLLING = auto()
    GREEDY_UNTIL = auto()
    GREEDY_UNTIL_MULTI_TURN = auto()
    GREEDY_UNTIL_WITH_LOGITS = auto()


@dataclass
class Request:
    """
    Represents a request for a specific task, example and request within that
    example in the evaluation process.
    For example in the task "boolq", the example "Is the sun hot?" and the
    requests for that example "Is the sun hot? Yes" and "Is the sun hot? No".

    Attributes:
        task_name (str): The name of the task.
        example_index (int): The index of the example.
        request_index (int): The index of the request.
        context (str): The context for the request.
    """

    task_name: str
    example_index: int
    request_index: int
    context: str


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


@dataclass
class LoglikelihoodRollingRequest(Request):
    """
    Represents a request for log-likelihood rolling evaluation.

    Inherits from the base Request class.
    """

    request_type = RequestType.LOGLIKELIHOOD_ROLLING
    tokenized_context: list[int] = None
    tokenized_continuation: list[int] = None


@dataclass
class GreedyUntilRequest(Request):
    """
    Represents a request for generating text using the Greedy-Until algorithm.

    Attributes:
        stop_sequence (str): The sequence of tokens that indicates when to stop generating text.
        generation_size (int): The maximum number of tokens to generate.
        request_type (RequestType): The type of the request, set to RequestType.GREEDY_UNTIL.
    """

    stop_sequence: Union[str, tuple[str], list[str]]
    generation_size: int
    request_type = RequestType.GREEDY_UNTIL
    tokenized_context: list[int] = None


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


@dataclass
class GreedyUntilWithLogitsRequest(Request):
    """
    Represents a request for generating text using the Greedy-Until strategy but
    returning the logits.

    Attributes:
        stop_sequence (str): The sequence of tokens that indicates when to stop generating text.
        generation_size (int): The maximum number of tokens to generate.
        request_type (RequestType): The type of the request (GREEDY_UNTIL_WITH_LOGITS).
    """

    stop_sequence: Union[str, tuple[str], list[str]]
    generation_size: int
    request_type = RequestType.GREEDY_UNTIL_WITH_LOGITS
    tokenized_context: list[int] = None


class TaskExampleId(NamedTuple):
    """
    Represents the identifier for an example in a task.

    Attributes:
        task_name (str): The name of the task.
        doc_id_seed (str): The document id with the seed used for few_shot appended at the end.
    """

    task_name: str
    doc_id_seed: str


@dataclass(slots=True)
class Doc:
    """
    Dataclass used to represent the content of a task example
    almost every field is optional, but some tasks require some fields to be present
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
    instruction: Optional[list[str]] = None
    target_for_fewshot_sorting: Optional[str] = None  # will probably have to be removed in the future

    # Filled when parsing and adding the few-shot context
    ctx: Optional[str] = ""
    num_asked_few_shots: int = -1
    num_effective_few_shots: int = -1

    def get_golds(self, few_shot: bool = False):
        """Return gold targets extracted from the target dict"""
        gold_indices = as_list(self.gold_index)
        if few_shot and self.target_for_fewshot_sorting is not None:
            choices = self.target_for_fewshot_sorting
            if isinstance(choices, str):  # correct choice is already selected
                return choices
        else:
            choices = self.choices
        golds = []
        for gold_ix in gold_indices:
            local_golds = as_list(choices[gold_ix])
            for local_gold in local_golds:
                golds.append(local_gold)
        return golds

    def __repr__(self):
        doc_dict = asdict(self)
        return json.dumps(doc_dict)

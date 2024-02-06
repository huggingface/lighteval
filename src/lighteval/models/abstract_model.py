from abc import ABC, abstractmethod
from typing import Optional

from lighteval.models.model_config import EnvConfig
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    GreedyUntilWithLogitsRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)


class LightevalModel(ABC):
    """Abstract model class defining the API that every model to plug into lighteval must follow."""

    @abstractmethod
    def __init__(
        self,
        config,
        env_config: EnvConfig,
    ):
        self.tokenizer = None
        return NotImplemented

    def cleanup(self):
        return

    @property
    @abstractmethod
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        raise NotImplementedError

    def greedy_until_with_logits(
        self,
        requests: list[GreedyUntilWithLogitsRequest],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
        dataset_splits: int = 4,
    ) -> list[GenerateReturn]:
        """
        Generates sequences greedily until a stopping condition is met,
        returning both the generated sequences and the logits.

        Args:
            requests (list[tuple[str, dict]]): A list of input requests,
                where each request is a tuple containing a prompt string and a dictionary of additional parameters.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
            override_bs (Optional[int], optional): Overrides the batch size for generation. Defaults to None.
            dataset_splits (int, optional): Number of splits to divide the dataset into for parallel generation. Defaults to 4.

        Returns:
            list[GenerateReturn]: A list of GenerateReturn objects,
                where each object contains the generated sequence and the corresponding logits.
        """
        return self.greedy_until(
            requests=requests,
            disable_tqdm=disable_tqdm,
            override_bs=override_bs,
            dataset_splits=dataset_splits,
            returns_logits=True,
        )

    @abstractmethod
    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        returns_logits: bool = False,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
        dataset_splits: int = 4,
    ) -> list[GenerateReturn]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            returns_logits (bool, optional): Whether to return the logits of the generated responses. Defaults to False.
            disable_tqdm (bool, optional): Whether to disable the progress bar. Defaults to False.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.
            dataset_splits (int, optional): Number of splits to divide the dataset into. Defaults to 4.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        return NotImplemented

    @abstractmethod
    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        return NotImplemented

    @abstractmethod
    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs=None
    ) -> list[LoglikelihoodReturn]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        return NotImplemented

    @abstractmethod
    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenReturn]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        return NotImplemented

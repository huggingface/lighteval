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
import re
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import yaml
from pydantic import BaseModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


TokenSequence = Union[list[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class ModelConfig(BaseModel, extra="forbid"):
    """
    Base configuration class for all model types in Lighteval.

    This is the foundation class that all specific model configurations inherit from.
    It provides common functionality for parsing configuration from files and command-line arguments,
    as well as shared attributes that are used by all models like generation parameters and system prompts.

    Attributes:
        generation_parameters (GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc. Defaults to empty GenerationParameters.
        system_prompt (str | None):
            Optional system prompt to be used with chat models. This prompt sets the
            behavior and context for the model during evaluation.

    Methods:
        from_path(path: str):
            Load configuration from a YAML file.
        from_args(args: str):
            Parse configuration from a command-line argument string.
        _parse_args(args: str):
            Static method to parse argument strings into configuration dictionaries.

    Example:
        ```python
        # Load from YAML file
        config = ModelConfig.from_path("model_config.yaml")

        # Load from command line arguments
        config = ModelConfig.from_args("model_name=meta-llama/Llama-3.1-8B-Instruct,system_prompt='You are a helpful assistant.',generation_parameters={temperature=0.7}")

        # Direct instantiation
        config = ModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            generation_parameters=GenerationParameters(temperature=0.7),
            system_prompt="You are a helpful assistant."
        )
        ```
    """

    generation_parameters: GenerationParameters = GenerationParameters()
    system_prompt: str | None = None
    cache_dir: str = "~/.cache/huggingface/lighteval"

    @classmethod
    def from_path(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return cls(**config["model_parameters"])

    @classmethod
    def from_args(cls, args: str):
        config = cls._parse_args(args)
        return cls(**config)

    @staticmethod
    def _parse_args(args: str) -> dict:
        """Parse a string of arguments into a configuration dictionary.

        This function parses a string containing model arguments and generation parameters
        into a structured dictionary with two main sections: 'model' and 'generation'.
        It specifically handles generation parameters enclosed in curly braces.

        Args:
            args (str): A string containing comma-separated key-value pairs, where generation
                parameters can be specified in a nested JSON-like format.

        Returns:
            dict: A dictionary with two keys:
                - 'model': Contains general model configuration parameters
                - 'generation': Contains generation-specific parameters

        Examples:
            >>> parse_args("model_name=gpt2,max_length=100")
            {
                'model': {'model_name': 'gpt2', 'max_length': '100'},
            }

            >>> parse_args("model_name=gpt2,generation_parameters={temperature:0.7,top_p:0.9}")
            {
                'model': {'model_name': 'gpt2', 'generation_parameters': {'temperature': 0.7, 'top_p': 0.9},
            }

            >>> parse_args("model_name=gpt2,use_cache,generation_parameters={temperature:0.7}")
            {
                'model': {'model_name': 'gpt2', 'use_cache': True, 'generation_parameters': {'temperature': 0.7}},
            }
        """
        # Looking for generation_parameters in the model_args
        generation_parameters_dict = None
        pattern = re.compile(r"(\w+)=(\{.*\}|[^,]+)")
        matches = pattern.findall(args)
        for key, value in matches:
            key = key.strip()
            if key == "generation_parameters":
                # Keys must be quoted (since they are strings)
                gen_params = re.sub(r"(\w+):", r'"\1":', value)
                # for k, v where v are strings, we quote them too
                gen_params = re.sub(r":\s*([A-Za-z_][\w.-]*)\s*(?=[,}])", r':"\1"', gen_params)
                generation_parameters_dict = json.loads(gen_params)

        args = re.sub(r"generation_parameters=\{.*?\},?", "", args).strip(",")
        model_config = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in args.split(",")}

        if generation_parameters_dict is not None:
            model_config["generation_parameters"] = generation_parameters_dict

        return model_config


class LightevalModel(ABC):
    DATASET_SPLITS = 4
    is_async = False

    """Abstract model class defining the API that every model to plug into lighteval must follow."""

    def cleanup(self):
        """Clean up operations if needed, such as closing an endpoint."""
        return

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError

    @property
    @abstractmethod
    def add_special_tokens(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        raise NotImplementedError

    @property
    def disable_tqdm(self) -> bool:
        return False

    @abstractmethod
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        return NotImplemented

    @abstractmethod
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        return NotImplemented

    @abstractmethod
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        return NotImplemented

    # Tokenization utils
    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None) -> TokenSequence:
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        if isinstance(str_to_encode, str):
            return self.tokenizer.encode(str_to_encode, add_special_tokens=add_special_tokens)
        return self.tokenizer(
            str_to_encode,
            padding=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

    def tok_encode_pair(self, context, continuations: list[str], pairwise: bool = False):
        """Encodes a context with a list of continuations by taking care of the spaces in between.
        Args:
            context (str): The context string to be encoded.
            continuation (list[str]): List of continuation strings to be encoded.
            pairwise (bool):
                If True, encode context and continuations separately.
                If False, encode them together and then split.

        Returns:
            Tuple[TokenSequence, list[TokenSequence]]:
                A tuple containing the encoded context and a list of encoded continuations.

        The advantage of pairwise is:
        1) It better aligns with how LLM predicts tokens
        2) Works in case len(tok(context,cont)) != len(tok(context)) + len(tok(continuation)).
        E.g this can happen for chinese if no space is used between context/continuation
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuations = [context[-n_spaces:] + cont for cont in continuations]
            context = context[:-n_spaces]

        if pairwise:
            # We don't add special tokens to the continuation as if bos is added
            # models tend to to completely ignore a context
            context_enc = self.tok_encode(context, add_special_tokens=self.add_special_tokens)
            continuation_enc = [self.tok_encode(cont, add_special_tokens=False) for cont in continuations]

            # In theory the context_enc can be ended with eos token, this would again
            # cause the model to ignore the context. We thus strip the eos token from context_enc
            if len(context_enc) > 0 and context_enc[-1] == self.tokenizer.eos_token_id:
                context_enc = context_enc[:-1]

            context_encs = [context_enc] * len(continuation_enc)

            return context_encs, continuation_enc

        # Handle list of continuations
        context_enc = self.tok_encode(context)
        context_encs = []
        continuations_encs = []
        for cont in continuations:
            whole_enc = self.tok_encode(context + cont)
            context_enc_len = len(context_enc)
            if len(context_enc) == len(whole_enc):
                context_enc_len = len(context_enc) - 1
            continuations_encs.append(whole_enc[context_enc_len:])
            context_encs.append(whole_enc[:context_enc_len])

        return context_encs, continuations_encs

    def tok_decode(self, tokens: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

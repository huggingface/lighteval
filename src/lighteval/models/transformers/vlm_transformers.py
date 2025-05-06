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

import logging
from typing import Union

import torch
from pydantic import PositiveInt
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.models.utils import ModelConfig, _get_model_sha, _simplify_name
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.imports import (
    is_accelerate_available,
)


logger = logging.getLogger(__name__)


if is_accelerate_available():
    from datetime import timedelta

    from accelerate import Accelerator, InitProcessGroupKwargs


class VLMTransformersModelConfig(ModelConfig):
    """
    Base configuration class for models.

    Attributes:
        model_name (str):
            HuggingFace Hub model ID name or the path to a pre-trained
            model to load. This is effectively the `pretrained_model_name_or_path`
            argument of `from_pretrained` in the HuggingFace `transformers` API.
        accelerator (Accelerator): accelerator to use for model training.
        tokenizer (Optional[str]): HuggingFace Hub tokenizer ID that will be
            used for tokenization.
        multichoice_continuations_start_space (Optional[bool]): Whether to add a
            space at the start of each continuation in multichoice generation.
            For example, context: "What is the capital of France?" and choices: "Paris", "London".
            Will be tokenized as: "What is the capital of France? Paris" and "What is the capital of France? London".
            True adds a space, False strips a space, None does nothing
        pairwise_tokenization (bool): Whether to tokenize the context and continuation as separately or together.
        subfolder (Optional[str]): The subfolder within the model repository.
        revision (str): The revision of the model.
        batch_size (int): The batch size for model training.
        max_gen_toks (Optional[int]): The maximum number of tokens to generate.
        max_length (Optional[int]): The maximum length of the generated output.
        add_special_tokens (bool, optional, defaults to True): Whether to add special tokens to the input sequences.
           If `None`, the default value will be set to `True` for seq2seq models (e.g. T5) and
            `False` for causal models.
        model_parallel (bool, optional, defaults to None):
            True/False: force to use or not the `accelerate` library to load a large
            model across multiple devices.
            Default: None which corresponds to comparing the number of processes with
                the number of GPUs. If it's smaller => model-parallelism, else not.
        dtype (Union[str, torch.dtype], optional, defaults to None):):
            Converts the model weights to `dtype`, if specified. Strings get
            converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
            Use `dtype="auto"` to derive the type from the model's weights.
        device (Union[int, str]): device to use for model training.
        quantization_config (Optional[BitsAndBytesConfig]): quantization
            configuration for the model, manually provided to load a normally floating point
            model at a quantized precision. Needed for 4-bit and 8-bit precision.
        trust_remote_code (bool): Whether to trust remote code during model
            loading.
        generation_parameters (GenerationParameters): Range of parameters which will affect the generation.
        generation_config (GenerationConfig): GenerationConfig object (only passed during manual creation)

    Methods:
        __post_init__(): Performs post-initialization checks on the configuration.
        _init_configs(model_name, env_config): Initializes the model configuration.
        init_configs(env_config): Initializes the model configuration using the environment configuration.
        get_model_sha(): Retrieves the SHA of the model.

    """

    model_name: str
    tokenizer: str | None = None
    subfolder: str | None = None
    revision: str = "main"
    batch_size: PositiveInt | None = None
    generation_size: PositiveInt = 256
    max_length: PositiveInt | None = None
    add_special_tokens: bool = True
    model_parallel: bool | None = None
    dtype: str | None = None
    device: Union[int, str] = "cuda"
    trust_remote_code: bool = False
    use_chat_template: bool = False
    compile: bool = False
    pairwise_tokenization: bool = False
    device_map: str | None = None

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.model_name, revision=self.revision)


class VLMTransformersModel(LightevalModel):
    def __init__(
        self,
        config: VLMTransformersModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self.config = config
        self.accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
        self._device = self.accelerator.device
        self.use_chat_template = config.use_chat_template
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space
        self._add_special_tokens = config.add_special_tokens or False
        self.pairwise_tokenization = config.pairwise_tokenization
        self.batch_size = config.batch_size
        self.transformers_config = config.get_transformers_config()

        self.model_sha = config.get_model_sha()
        self._max_length = self._init_max_length()
        self._processor = self._create_auto_processor()
        self.model = self._create_auto_model()

        # We are in DP (and launch the script with `accelerate launch`)
        if config.model_parallel is False and self.config.dtype not in ["4bit", "8bit"]:
            logger.info(f"Using Data Parallelism, putting model on device {self._device}")
            self.model = self.model.to(self._device)
        if config.compile:
            try:
                logger.info("Compiling the model")
                self.model.model.compile()
            except AttributeError as e:
                logger.warning("Could not compile the model because: ", e)

        self.model_name = _simplify_name(config.model_name)

        self.generation_config_dict = config.generation_parameters.to_transformers_dict()

        self.model_info = ModelInfo(
            model_name=self.config.model_name,
            model_sha=self.model_sha,
            model_dtype=config.dtype,
        )

    @property
    def tokenizer(self):
        return self._processor

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    @property
    def disable_tqdm(self) -> bool:
        disable_tqdm = False
        if self.accelerator:
            disable_tqdm = bool(not self.accelerator.is_main_process)
        return disable_tqdm

    def _create_auto_model(self):
        subfolder = self.config.subfolder
        revision = self.config.revision + (f"/{subfolder}" if subfolder is not None else "")

        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name,
            revision=revision,
            device_map="auto",  # TODO: self.config.device_map,
            torch_dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        model.eval()
        torch.set_grad_enabled(False)

        if self.config.compile:
            try:
                logger.info("Compiling the model")
                model.compile()
            except AttributeError as e:
                logger.warning("Could not compile the model because: ", e)

        return model

    def _create_auto_processor(self):
        """
        Create a Hugging Face AutoTokenizer for language model.

        Returns:
            transformers.PreTrainedTokenizer: The created tokenizer.
        """
        tokenizer_name = self.config.tokenizer or self.config.model_name
        subfolder = self.config.subfolder
        revision = self.config.revision + (f"/{subfolder}" if subfolder is not None else "")

        processor = AutoProcessor.from_pretrained(
            tokenizer_name,
            revision=revision,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",
            truncation_side="left",
        )

        return processor

    def _init_max_length(self) -> int:
        """Return the maximum sequence length of the model.

        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.

        Returns:
            int: Max length to use depending on the available args and config
        """
        if self.config.max_length is not None:
            return self.config.max_length

        # Try to get the sequence length from the model config. It's no super robust
        text_model_config = self.transformers_config.get_text_config()
        max_seq_length = getattr(text_model_config, "max_position_embeddings", None)
        if max_seq_length is not None:
            return max_seq_length

        logger.warning(
            "No max_length attribute found in the model config. Using the default max sequence length setting `2048`. "
            "It is recommended to set max_length trough the model args: max_length=..."
        )

        return 2048

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        results = []

        for request in tqdm(requests, desc="Generating responses"):
            texts = request.context
            images = request.specific["images"]

            inputs = self._processor(
                text=texts,
                images=images,
                return_tensors="pt",
            )
            inputs = inputs.to(self._device)

            outputs = self.model.generate(**inputs, return_dict_in_generate=False)
            input_ids = inputs.input_ids
            generated_ids = outputs[:, input_ids.shape[1] :]
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_response = GenerativeResponse(
                result=generated_text,
                generated_tokens=generated_ids,
                input_tokens=input_ids,
                truncated_tokens_count=0,
                padded_tokens_count=0,
            )
            results.append(generated_response)

        return results

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
    ) -> list[LoglikelihoodResponse]:
        raise NotImplementedError()

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest]
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            requests (list[Tuple[str, dict]]): _description_

        Returns:
            list[Tuple[float, bool]]: _description_
        """
        raise NotImplementedError()

    def loglikelihood_rolling(
        self,
        requests: list[LoglikelihoodRequest],
    ) -> list[LoglikelihoodResponse]:
        raise NotImplementedError()

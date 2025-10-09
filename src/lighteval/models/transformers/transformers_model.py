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
import os
from datetime import timedelta
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import calculate_maximum_sizes, convert_bytes, get_max_memory
from pydantic import Field, PositiveInt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import (
    Batch,
    ModelResponse,
)
from lighteval.models.utils import _get_dtype, _get_model_sha, _simplify_name, uses_chat_template
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import (
    is_package_available,
)
from lighteval.utils.parallelism import find_executable_batch_size


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class TransformersModelConfig(ModelConfig):
    """Configuration class for HuggingFace Transformers models.

    This configuration is used to load and configure models from the HuggingFace Transformers library.

    Attributes:
        model_name (str):
            HuggingFace Hub model ID or path to a pre-trained model. This corresponds to the
            `pretrained_model_name_or_path` argument in HuggingFace's `from_pretrained` method.
        tokenizer (str | None):
            Optional HuggingFace Hub tokenizer ID. If not specified, uses the same ID as model_name.
            Useful when the tokenizer is different from the model (e.g., for multilingual models).
        subfolder (str | None):
            Subfolder within the model repository. Used when models are stored in subdirectories.
        revision (str):
            Git revision of the model to load. Defaults to "main".
        batch_size (PositiveInt | None):
            Batch size for model inference. If None, will be automatically determined.
        max_length (PositiveInt | None):
            Maximum sequence length for the model. If None, uses model's default.
        model_loading_kwargs (dict):
            Additional keyword arguments passed to `from_pretrained`. Defaults to empty dict.
        add_special_tokens (bool):
            Whether to add special tokens during tokenization. Defaults to True.
        skip_special_tokens (bool):
            Whether the tokenizer should output special tokens back during generation. Needed for reasoning models. Defaults to True
        model_parallel (bool | None):
            Whether to use model parallelism across multiple GPUs. If None, automatically
            determined based on available GPUs and model size.
        dtype (str | None):
            Data type for model weights. Can be "float16", "bfloat16", "float32", "auto", "4bit", "8bit".
            If "auto", uses the model's default dtype.
        device (Union[int, str]):
            Device to load the model on. Can be "cuda", "cpu", or GPU index. Defaults to "cuda".
        trust_remote_code (bool):
            Whether to trust remote code when loading models. Defaults to False.
        compile (bool):
            Whether to compile the model using torch.compile for optimization. Defaults to False.
        multichoice_continuations_start_space (bool | None):
            Whether to add a space before multiple choice continuations. If None, uses model default.
            True forces adding space, False removes leading space if present.
        pairwise_tokenization (bool):
            Whether to tokenize context and continuation separately or together. Defaults to False.
        continuous_batching (bool):
            Whether to use continuous batching for generation. Defaults to False.
        override_chat_template (bool):
            If True, we force the model to use a chat template. If alse, we prevent the model from using
            a chat template. If None, we use the default (true if present in the tokenizer, false otherwise)
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.

    Example:
        ```python
        config = TransformersModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=4,
            dtype="float16",
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```

    Note:
        This configuration supports quantization (4-bit and 8-bit) through the dtype parameter.
        When using quantization, ensure you have the required dependencies installed
        (bitsandbytes for 4-bit/8-bit quantization).
    """

    model_name: str
    tokenizer: str | None = None
    subfolder: str | None = None
    revision: str = "main"
    batch_size: PositiveInt | None = None
    max_length: PositiveInt | None = None
    model_loading_kwargs: dict = Field(default_factory=dict)
    add_special_tokens: bool = True
    skip_special_tokens: bool = True
    model_parallel: bool | None = None
    dtype: str | None = None
    device: Union[int, str] = "cuda"
    trust_remote_code: bool = False
    compile: bool = False
    multichoice_continuations_start_space: bool | None = None
    pairwise_tokenization: bool = False
    continuous_batching: bool = False
    override_chat_template: bool = None

    def model_post_init(self, __context):
        if self.multichoice_continuations_start_space is True:
            logger.warning(
                "You set `multichoice_continuations_start_space` to true. This will force multichoice continuations to use a starting space"
            )
        if self.multichoice_continuations_start_space is False:
            logger.warning(
                "You set `multichoice_continuations_start_space` to false. This will remove a leading space from multichoice continuations, if present."
            )

    def get_transformers_config(self) -> PretrainedConfig:
        revision = self.revision

        if self.subfolder:
            revision = f"{self.revision}/{self.subfolder}"

        auto_config = AutoConfig.from_pretrained(
            self.model_name,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
        )

        return auto_config

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.model_name, revision=self.revision)


class TransformersModel(LightevalModel):
    def __init__(
        self,
        config: TransformersModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self.config = config
        self.accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
        self._device = self.accelerator.device
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space
        self._add_special_tokens = config.add_special_tokens or False
        self.skip_special_tokens = config.skip_special_tokens or True
        self.pairwise_tokenization = config.pairwise_tokenization
        self.batch_size = config.batch_size
        self.continuous_batching = config.continuous_batching
        self.transformers_config = config.get_transformers_config()
        self.generation_config_dict = config.generation_parameters.to_transformers_dict()

        self.model_sha = config.get_model_sha()
        self._max_length = self._init_max_length()
        self._tokenizer = self._create_auto_tokenizer()
        self.use_chat_template = uses_chat_template(
            tokenizer=self._tokenizer, override_chat_template=config.override_chat_template
        )
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

        if is_package_available("accelerate"):
            model_size, _ = calculate_maximum_sizes(self.model)
            model_size = convert_bytes(model_size)
        else:
            model_size = -1

        self.prompt_manager = PromptManager(
            use_chat_template=self.use_chat_template, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    def cleanup(self):
        """Clean up operations if needed, such as closing an endpoint."""
        del self.model
        del self._tokenizer
        torch.cuda.empty_cache()

    @classmethod
    def from_model(
        cls,
        model: AutoModelForCausalLM,
        config: TransformersModelConfig,
        accelerator: Accelerator | None = None,
    ) -> "TransformersModel":
        if config is None:
            raise ValueError("Config must be provided to initialize the TransformersModel via `from_model` method.")

        # Instanciate the object without using __init__
        self = cls.__new__(cls)

        self.transformers_config = model.config

        self.config = config
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space
        self._add_special_tokens = config.add_special_tokens
        self.skip_special_tokens = config.skip_special_tokens
        self.pairwise_tokenization = config.pairwise_tokenization
        self.batch_size = config.batch_size
        self.continuous_batching = config.continuous_batching
        self.generation_config_dict = config.generation_parameters.to_transformers_dict()

        self.model_name = config.model_name
        self.model_sha = config.get_model_sha()
        self._max_length = self._init_max_length()
        self._tokenizer = self._create_auto_tokenizer()
        self.use_chat_template = uses_chat_template(
            tokenizer=self._tokenizer, override_chat_template=config.override_chat_template
        )

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)

        self.accelerator = accelerator
        if accelerator is not None:
            self._device = accelerator.device
            self.model = self.accelerator.prepare(self.model.to(accelerator.device))
        else:
            self._device = self.config.device

        if is_package_available("accelerate"):
            model_size, _ = calculate_maximum_sizes(self.model)
            model_size = convert_bytes(model_size)
        else:
            model_size = -1
        self.prompt_manager = PromptManager(
            use_chat_template=self.use_chat_template,
            tokenizer=self.tokenizer,
            system_prompt=config.system_prompt if config else None,
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config) if config else None

        return self

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def device(self) -> Union[str, torch.device]:
        return self._device

    @property
    def disable_tqdm(self) -> bool:
        disable_tqdm = False
        if self.accelerator:
            disable_tqdm = bool(not self.accelerator.is_main_process)
        return disable_tqdm

    def init_model_parallel(self, model_parallel: bool | None = None) -> Tuple[bool, Optional[dict], Optional[str]]:
        """Compute all the parameters related to model_parallel"""
        if not is_package_available("accelerate"):
            return False, None, None

        self.num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.num_machines = torch.cuda.device_count() // self.num_local_processes

        if self.num_machines == 1:
            logger.info("We are not in a distributed setting. Setting model_parallel to False.")
            model_parallel = False

        if model_parallel is None:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            model_parallel = bool(self.num_local_processes < len(max_memory_all_gpus))
            logger.info(
                f"Setting model parallel to {model_parallel} since "
                f"the number of local processes is {self.num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )
        if model_parallel is True:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_mem_this_process = {
                k: v
                for k, v in max_memory_all_gpus.items()
                if k % self.num_local_processes == (self.accelerator.process_index % self.num_local_processes)
            }
            device_map = "auto"
            logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_mem_this_process} and device map to {device_map}"
            )
        else:
            max_mem_this_process = None
            device_map = None
            logger.info(
                f"Model parallel was set to False, max memory set to {max_mem_this_process} and device map to {device_map}"
            )
        return model_parallel, max_mem_this_process, device_map

    def _create_auto_model(self) -> transformers.PreTrainedModel:
        """Creates an instance of the pretrained HF model.

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        model_parallel, max_memory, device_map = self.init_model_parallel(self.config.model_parallel)
        self.config.model_parallel = model_parallel

        if self.config.dtype == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif self.config.dtype == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        torch_dtype = _get_dtype(self.config.dtype)
        subfolder = self.config.subfolder
        revision = self.config.revision + (f"/{subfolder}" if subfolder is not None else "")

        pretrained_config = self.transformers_config

        kwargs = self.config.model_loading_kwargs.copy()
        if "quantization_config" not in pretrained_config.to_dict():
            kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=revision,
            max_memory=max_memory,
            device_map=device_map,
            # tp_plan="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            **kwargs,
        )
        # model.to(self.device)
        model.eval()
        torch.set_grad_enabled(False)
        if self.continuous_batching:
            generation_config = GenerationConfig(
                **self.generation_config_dict,
            )
            model.generation_config = generation_config

        if self.config.compile:
            try:
                logger.info("Compiling the model")
                model.compile()
            except AttributeError as e:
                logger.warning("Could not compile the model because: ", e)

        return model

    def _create_auto_tokenizer(
        self,
    ) -> transformers.PreTrainedTokenizer:
        """Create a Hugging Face AutoTokenizer for language model.

        Returns:
            transformers.PreTrainedTokenizer: The created tokenizer.
        """
        tokenizer_name = self.config.tokenizer or self.config.model_name
        subfolder = self.config.subfolder
        revision = self.config.revision + (f"/{subfolder}" if subfolder is not None else "")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=revision,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",
            truncation_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self.max_length
        logger.info("Tokenizer truncation and padding size set to the left side.")

        return tokenizer

    def _init_max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.

        Returns:
            int: Max length to use depending on the available args and config
        """
        if self.config.max_length is not None:
            return self.config.max_length

        # Try to get the sequence length from the model config.
        text_model_config = self.transformers_config.get_text_config()

        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(text_model_config, attr):
                return getattr(text_model_config, attr)

        logger.warning(
            "No max_length attribute found in the model config. Using the default max sequence length setting `2048`. "
            "It is recommended to set max_length trough the model args: max_length=..."
        )

        return 2048

    def _check_continuations_start_space(self, continuation: str) -> str:
        """Some models tokenizer want a space at the beginning and other not. We update this if needed here.
        multichoice_continuations_start_space can be:
        - True (add a space if these isn't one)
        - False (remove a space if there is one)
        - None (Don't touch - default)
        Todo: find a way to add this back WITHOUT breaking compatibility with the harness
        """
        if self.multichoice_continuations_start_space is not None:
            if self.multichoice_continuations_start_space and continuation[0] != " ":
                continuation = " " + continuation
            if not self.multichoice_continuations_start_space and continuation[0] == " ":
                continuation = continuation.lstrip()
        return continuation

    def _model_call(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).logits

    def _get_batch_size(self, max_input_length: int, override_bs: int | None, starting_batch_size: int = 512) -> int:
        if override_bs is not None:
            return override_bs
        logger.info(f"Detecting largest batch size with max_input_length={max_input_length}")

        @find_executable_batch_size(
            starting_batch_size=starting_batch_size
        )  # if OOM, then halves batch_size and tries again
        def forward_batch(batch_size):
            test_batch = torch.ones(
                (batch_size + int(0.1 * batch_size), max_input_length), device=self.device
            ).long()  # We add 10% for marging :)
            F.log_softmax(self._model_call(test_batch).float(), dim=-1).cpu()
            return batch_size

        batch_size = forward_batch()
        logger.info(f"Determined largest batch size: {batch_size}")
        return batch_size

    def _continuous_greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
            if self.use_chat_template:
                stop_tokens = []
            else:
                # NOTE: we are assuming all items in a batch behave similarly (same
                # stop_tokens and max_tokens genrated) which is not necessarily
                # the case! Because of that we only use batch size of 1
                stop_tokens = split[0].stop_sequence

            max_new_tokens = self.config.generation_parameters.max_new_tokens or split[0].generation_size
            returns_logits = split[0].use_logits
            num_samples = split[0].num_samples
            contexts = [self.prompt_manager.prepare_prompt(doc) for doc in split]
            tokenized = self.tokenizer(contexts, add_special_tokens=self.add_special_tokens)

            # The main question for this step is the following:
            # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
            # of losing some meaning, or have some generations that are exceedingly short?
            # The choice we go for here is to avoid truncating the prompt if we can, since it
            # should have been managed by the prompt creator/few shot manager if requested by the user.
            inputs = tokenized["input_ids"]
            context_size = len(inputs[0])

            # left truncate the inputs to the maximum length
            if max_new_tokens is not None:
                if context_size + max_new_tokens > self.max_length:
                    logger.warning(
                        f"{context_size + max_new_tokens=} which is greater than {self.max_length=}. Truncating context to {self.max_length - max_new_tokens} tokens."
                    )
                    context_size = self.max_length - max_new_tokens
                    if context_size < 0:
                        logger.critical(
                            f"{context_size=} is less than 0, either reduce the max_new_tokens or increase model max length."
                        )
                        raise ValueError("Context size is less than 0.")
                    inputs = [input[-context_size:] for input in inputs]
            else:
                if context_size > self.max_length:
                    logger.warning(
                        f"{context_size=} which is greater than {self.max_length=}. Truncating context to {self.max_length} tokens."
                    )
                    context_size = self.max_length
                    inputs = [input[-context_size:] for input in inputs]

            _outputs = self._generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                returns_logits=returns_logits,
                num_samples=num_samples,
                continuous_batching=True,
            )

            for req_id, _output in _outputs.items():
                output_token_ids = []
                logprobs_raw = []
                result = []

                # for output in _output.outputs:
                output_token_ids.append(_output.generated_tokens)
                # logprobs_raw.append(output.logprobs)
                result.append(
                    self.tokenizer.decode(_output.generated_tokens, skip_special_tokens=self.skip_special_tokens)
                )

                if logprobs_raw and output_token_ids and False:
                    logprobs = [logprobs_raw[0][token_id].logprob for token_id in output_token_ids[0]]
                else:
                    logprobs = []

                input_token_ids = _output.prompt_ids
                cur_response = ModelResponse(
                    text=result,
                    logprobs=logprobs,
                    output_tokens=output_token_ids,
                    input_tokens=input_token_ids,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    def _padded_greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        results = []
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            if split[0].generation_size is None:
                # No constraints on the generation size: max length allowed is the max model context
                max_context_continuation_size_allowed = self.max_length
            else:
                context = self.prompt_manager.prepare_prompt(split[0])
                tokenized_context = self.tokenizer(context)

                # Longest context in the current split is the first item (since we sort reversed)
                longest_context_continuation_size_in_split = (
                    len(tokenized_context["input_ids"]) + split[0].generation_size
                )
                max_context_continuation_size_allowed = min(
                    longest_context_continuation_size_in_split, self.max_length
                )
            batch_size = self._get_batch_size(
                override_bs=self.config.batch_size,
                max_input_length=max_context_continuation_size_allowed,
                starting_batch_size=starting_batch_size,
            )
            # For next iteration, since the batch will be smaller, we'll test a bigger batch size
            starting_batch_size = batch_size * 2

            dataloader = DataLoader(split, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                contexts = [self.prompt_manager.prepare_prompt(doc) for doc in batch]

                # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
                if self.use_chat_template:
                    stop_tokens = [self.tokenizer.eos_token]
                else:
                    # NOTE: we are assuming all items in a batch behave similarly (same
                    # stop_tokens and max_tokens genrated) which is not necessarily
                    # the case! Because of that we only use batch size of 1
                    stop_tokens = [self.tokenizer.eos_token] + batch[0].stop_sequences

                max_new_tokens = batch[0].generation_size
                num_samples = batch[0].num_samples

                # See doc https://huggingface.co/docs/transformers/v4.38.2/en/pad_truncation#padding-and-truncation
                # Will do left truncation and padding, as defined when creating the tokenizer
                tokenized = self.tokenizer(
                    contexts,
                    truncation="longest_first",  # we truncate to the model max length if needed
                    padding="longest",  # we pad to the longest sequence
                    return_tensors="pt",
                    max_length=max_context_continuation_size_allowed,  # we always allow minimum one token of generation
                    add_special_tokens=self.add_special_tokens,
                ).to(self.device)

                # The main question for this step is the following:
                # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
                # of losing some meaning, or have some generations that are exceedingly short?
                # The choice we go for here is to avoid truncating the prompt if we can, since it
                # should have been managed by the prompt creator/few shot manager if requested by the user.
                context_size = tokenized["input_ids"].shape[1]
                if context_size > self.max_length:
                    logger.warning(
                        f"The context size of your batch ({context_size}) is bigger than the maximum context size allowed by the model ({self.max_length}) for a task in"
                        + str({i.task_name for i in batch})
                        + ". This is likely to lead to some errors."  # noqa C401
                    )
                    # There will be truncation of at least one sample, maximum generation size will be one
                    max_new_tokens = 1
                else:  # We can't allow generation of more than max_length
                    if max_new_tokens is None:  # If generation size is not set, we go all the way
                        max_new_tokens = self.max_length - context_size
                    else:
                        max_new_tokens = min(self.max_length - context_size, max_new_tokens)
                        if max_new_tokens < 1:
                            max_new_tokens = 1

                prepared_batch = Batch(
                    input_ids=tokenized["input_ids"],
                    input_lengths=[len(item == 1) for item in tokenized["attention_mask"]],
                    input_mask=tokenized["attention_mask"],
                    truncated=[max(len(c) - tokenized["input_ids"].shape[1], 0) for c in contexts],
                    padded=[sum(mask == 0) for mask in tokenized["attention_mask"]],
                )

                cur_reponses = self._generate(
                    batch=prepared_batch,
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens,
                    returns_logits=False,
                    num_samples=num_samples,
                    continuous_batching=False,
                )
                results.extend(cur_reponses)

        return dataset.get_original_order(results)

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        if self.continuous_batching:
            return self._continuous_greedy_until(docs)
        else:
            return self._padded_greedy_until(docs)

    def _generate_continuous(
        self,
        inputs: list[list[int]],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
    ) -> Dict[str, ModelResponse]:
        # Compute model generation
        self.model.generation_config.use_cuda_graph = False  # Disable CUDA graph for batch generation
        self.model.generation_config.max_batch_tokens = 256  # Disable CUDA graph for batch generation
        # self.model.generation_config.do_sample = False  # Disable CUDA graph for batch generation
        batch_outputs = self.model.generate_batch(
            inputs=inputs,
            generation_config=self.model.generation_config,
            # You can pass request-specific overrides here, e.g., max_new_tokens=100
        )

        return batch_outputs

    def _generate_padded(
        self,
        batch: Batch,
        max_new_tokens: int,
        stop_tokens: list[str],
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
    ) -> list[ModelResponse]:
        """Contains the actual logic of the generation.
        First computes the stop sequences, then generates the predictions, then converts the outputs to ModelResponse.
        """
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop_sequences=stop_tokens, batch=batch)
        batch_size, _ = batch.input_ids.shape

        if num_samples > 1 and self.generation_config_dict["temperature"] == 0:
            raise ValueError(
                "You cannot generate multiple samples with temperature=0. Please set temperature > 0. Or use a non sampling metric."
            )

        generation_config = self.generation_config_dict.copy()
        generation_config.update(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=num_samples,
            output_logits=returns_logits,
            renormalize_logits=True,
        )
        if num_samples == 1 and generation_config["temperature"] == 0:
            generation_config["do_sample"] = False
        if num_samples > 1 and generation_config["temperature"] == 0:
            logger.warning("num_samples > 1 but temperature is set to 0, this will not sample different outputs.")

        # Compute model generation
        outputs: GenerateOutput = self.model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.input_mask,
            stopping_criteria=stopping_criteria,
            **generation_config,
        )
        generations = outputs.sequences[:, batch.input_ids.size(1) :]
        generations = torch.reshape(generations, (batch_size, num_samples, -1))
        generations, len_gens = self.pad_and_gather(generations, num_samples=num_samples)
        batch.input_ids, len_ids = self.pad_and_gather(batch.input_ids)

        logits, len_logits = None, None
        if returns_logits:
            logits, len_logits = self.pad_and_gather(outputs.logits)
            logits = logits.cpu().numpy()

        # We gather remaining info
        batch.truncated = torch.tensor(batch.truncated, device=self.device)
        if self.accelerator:
            batch.truncated = self.accelerator.gather_for_metrics(batch.truncated)
        batch.padded = torch.tensor(batch.padded, device=self.device)
        if self.accelerator:
            batch.padded = self.accelerator.gather_for_metrics(batch.padded)

        # We convert to ModelResponse outputs
        all_responses = []
        for ix, (batched_generations, batched_input, trunc, padded) in enumerate(
            zip(generations, batch.input_ids, batch.truncated, batch.padded)
        ):
            result_generations = []
            decoded_generations = []
            # Ensure the generated responses do not contain the stop sequences.
            for generation in batched_generations:
                generation = generation[: len_gens[ix]]
                result_generations.append(generation)
                decoded_generation = self.tok_decode([generation])[0]

                for term in stop_tokens:
                    decoded_generation = decoded_generation.split(term)[0]

                decoded_generations.append(decoded_generation)

            cur_response = ModelResponse(
                text=decoded_generations,
                output_tokens=result_generations,
                logits=logits[ix][: len_logits[ix]] if returns_logits else None,
                input_tokens=batched_input[: len_ids[ix]].tolist(),
                truncated_tokens_count=trunc.cpu().item(),
                padded_tokens_count=padded.cpu().item(),
            )
            all_responses.append(cur_response)

        return all_responses

    def _generate(
        self,
        continuous_batching: bool,
        **kwargs,
    ) -> list[ModelResponse]:
        if continuous_batching:
            return self._generate_continuous(**kwargs)
        else:
            return self._generate_padded(**kwargs)

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            docs (list[Doc]): List of documents containing the context and choices.

        Returns:
            list[ModelResponse]: List of model responses with logprobs.
        """
        return self._loglikelihood_tokens(docs)

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        return self._loglikelihood_tokens(
            docs,
            rolling=True,
        )

    def _loglikelihood_tokens(  # noqa: C901
        self,
        docs: list[Doc],
        rolling=False,
    ) -> list[ModelResponse]:
        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=1)
        starting_batch_size = STARTING_BATCH_SIZE
        all_responses = []

        for split in tqdm(dataset.splits_iterator(), disable=self.disable_tqdm):
            first_doc_context = self.prompt_manager.prepare_prompt(split[0])
            tokenized_contexts, tokenized_continuations = self.tok_encode_pair(
                first_doc_context, split[0].choices, pairwise=self.pairwise_tokenization
            )

            if rolling:  # we take all the sequence in rolling mode
                max_input_length = len(tokenized_contexts[0] + tokenized_continuations[0])
            else:  # in normal mode, we left cut the context if needed
                max_input_length = max(
                    min(self.max_length, len(tokenized_contexts[0] + tokenized_continuations[0]) - 1), 1
                )

            batch_size = self._get_batch_size(
                override_bs=self.config.batch_size,
                max_input_length=max_input_length,
                starting_batch_size=starting_batch_size,
            )
            starting_batch_size = batch_size * 2
            max_num_choices = max(len(d.choices) for d in split)
            # We divide the batch size by the number of choices as batch is samples * num choices
            # then round up to closest 8 multiple
            batch_size = max(1, round(batch_size // max_num_choices / 8) * 8)
            logger.warning(
                f"batch size is set to {batch_size} (it should be understood as '{batch_size} times the maximum number of choices per sample, {max_num_choices}')"
            )

            dataloader = DataLoader(split, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm):
                batch_contexts: list[str] = [self.prompt_manager.prepare_prompt(doc) for doc in batch]
                batch_tokenized_contexts = []
                batch_tokenized_continuations = []

                for context, doc in zip(batch_contexts, batch):
                    doc_contexts, doc_continuations = self.tok_encode_pair(context, doc.choices, pairwise=True)
                    batch_tokenized_contexts.append(doc_contexts)
                    batch_tokenized_continuations.append(doc_continuations)

                prepared_batch = self.prepare_batch_logprob(
                    tokenized_contexts=batch_tokenized_contexts,
                    tokenized_continuations=batch_tokenized_continuations,
                    max_context=None,  # computed as model max length in the function
                )

                model_output = self._model_call(prepared_batch.input_ids)
                logits = F.log_softmax(model_output, dim=-1)  # [batch, sequence_length, vocab]

                flat_index = 0
                batch_logits_sums = []
                batch_max_equals = []
                batch_tokenized_contexts_processed = []
                batch_tokenized_continuations_processed = []
                batch_choices_counts = []

                # Flatten the logits to match the number of choices per sample
                for doc_idx, doc in enumerate(batch):
                    # Get the size of the corresponding nested list
                    num_choices = len(batch_tokenized_continuations[doc_idx])
                    doc_continuations = batch_tokenized_continuations[doc_idx]
                    # Extract the corresponding elements from flat_values
                    doc_logits = logits[flat_index : flat_index + num_choices]
                    doc_input_lengths = prepared_batch.input_lengths[flat_index : flat_index + num_choices]
                    # Move the index forward
                    flat_index += num_choices

                    doc_logits_sums = []
                    doc_max_equals = []
                    doc_continuation_tokens = []

                    for choice_logits, input_length, choice_continuation in zip(
                        doc_logits, doc_input_lengths, doc_continuations
                    ):
                        choice_continuation_tensor = torch.tensor(
                            choice_continuation, dtype=torch.long, device=self.device
                        )
                        continuation_length = len(choice_continuation_tensor)
                        if rolling:
                            choice_logits = choice_logits.unsqueeze(0).to(self.device)  # [1, seq, vocab]
                            choice_continuation_tensor = (
                                choice_continuation_tensor[:input_length].unsqueeze(0).to(self.device)
                            )  # [1, seq]
                        else:
                            choice_logits = (
                                choice_logits[input_length - continuation_length - 1 : input_length - 1]
                                .unsqueeze(0)
                                .to(self.device)
                            )
                            choice_continuation_tensor = choice_continuation_tensor.unsqueeze(0).to(
                                self.device
                            )  # [1, seq]

                        # Check if per-token argmax is exactly equal to continuation
                        greedy_tokens = choice_logits.argmax(dim=-1).to(self.device)
                        # Sometimes the continuation is longer than allowed by the model, we only look at the first tokens
                        is_exact_match = (greedy_tokens == choice_continuation_tensor).all().squeeze(0).to(self.device)

                        # Obtain log-probs at the corresponding continuation token indices
                        choice_logits = torch.gather(
                            choice_logits, 2, choice_continuation_tensor.unsqueeze(-1)
                        ).squeeze(-1)  # [1, seq]

                        # Answer: (log prob, is-exact-match)
                        doc_logits_sums.append(choice_logits.sum())
                        doc_max_equals.append(is_exact_match)
                        doc_continuation_tokens.append(choice_continuation_tensor.squeeze(0))  # [seq]

                    batch_logits_sums.append(torch.stack(doc_logits_sums))
                    batch_max_equals.append(torch.stack(doc_max_equals))
                    batch_tokenized_continuations_processed.append(
                        pad_sequence(doc_continuation_tokens, batch_first=True, padding_value=-1)
                    )
                    batch_tokenized_contexts_processed.append(
                        torch.tensor(batch_tokenized_contexts[doc_idx][0], dtype=torch.long, device=self.device)
                    )
                    batch_choices_counts.append(
                        torch.tensor(len(batch_tokenized_continuations[doc_idx]), dtype=torch.long, device=self.device)
                    )

                # Gather the results from all processes
                # need to gather logits_sum_batch, max_equals_batch and batch_cont_tokens_batch, contexts
                if self.accelerator:
                    # Convert lists to tensors for proper gathering
                    # Pad and stack the tensors to make them gatherable
                    shape_choices = [
                        choices.shape for choices in batch_tokenized_continuations_processed
                    ]  # num_choices * max len choices
                    num_choices_tensor = torch.tensor([shape[0] for shape in shape_choices], device=self.device)
                    len_choices_tensor = torch.tensor([shape[1] for shape in shape_choices], device=self.device)
                    gathered_num_choices = self.accelerator.gather_for_metrics(num_choices_tensor)
                    gathered_len_choices = self.accelerator.gather_for_metrics(len_choices_tensor)
                    max_num_choices = gathered_num_choices.max().item()
                    max_len_choices = gathered_len_choices.max().item()
                    len_context_tensor = torch.tensor(
                        [len(ctx) for ctx in batch_tokenized_contexts_processed], device=self.device
                    )
                    gathered_len_context = self.accelerator.gather_for_metrics(len_context_tensor)
                    max_len_context = gathered_len_context.max().item()

                    # 1d - Pad logits_sum and max_equals to same number of choices
                    padded_logits_sums = []
                    for logits_sum_doc in batch_logits_sums:
                        pad_amount = max_num_choices - len(logits_sum_doc)
                        padded = F.pad(logits_sum_doc, (0, pad_amount), value=-1)
                        padded_logits_sums.append(padded)

                    padded_max_equals = []
                    for max_equals_doc in batch_max_equals:
                        pad_amount = max_num_choices - len(max_equals_doc)
                        padded = F.pad(max_equals_doc, (0, pad_amount), value=False)
                        padded_max_equals.append(padded)

                    # 2d - Pad continuations to max number of choice and max length
                    padded_continuations = []
                    for cont_batch in batch_tokenized_continuations_processed:
                        pad_amount_num = max_num_choices - cont_batch.shape[0]
                        pad_amount_len = max_len_choices - cont_batch.shape[1]
                        padded = F.pad(cont_batch, (0, pad_amount_len, 0, pad_amount_num), value=-1)
                        padded_continuations.append(padded)

                    # 1d - Pad context to maximum context size
                    padded_contexts = []
                    for ctx_batch in batch_tokenized_contexts_processed:
                        pad_amount = max_len_context - len(ctx_batch)
                        padded = F.pad(ctx_batch, (0, pad_amount), value=-1)
                        padded_contexts.append(padded)

                    # Stack all tensors for gathering
                    stacked_logits_sums = torch.stack(padded_logits_sums)
                    stacked_max_equals = torch.stack(padded_max_equals)
                    stacked_continuations = torch.stack(padded_continuations)
                    stacked_contexts = torch.stack(padded_contexts)

                    # Gather the stacked tensors
                    gathered_logits_sums = self.accelerator.gather_for_metrics(stacked_logits_sums)
                    gathered_max_equals = self.accelerator.gather_for_metrics(stacked_max_equals)
                    gathered_continuations = self.accelerator.gather_for_metrics(stacked_continuations)
                    gathered_contexts = self.accelerator.gather_for_metrics(stacked_contexts)

                    # Convert back to lists for processing
                    batch_logits_sums = []
                    batch_max_equals = []
                    batch_tokenized_continuations_processed = []
                    batch_tokenized_contexts_processed = []

                    # Only process if we have gathered results
                    for i, num_choices in enumerate(gathered_num_choices):
                        # Extract non-padded values
                        # 1d on num choices
                        batch_logits_sums.append(gathered_logits_sums[i][:num_choices])
                        batch_max_equals.append(gathered_max_equals[i][:num_choices])
                        # 2d on num choices and max len
                        len_choice = gathered_len_choices[i]
                        batch_tokenized_continuations_processed.append(
                            gathered_continuations[i][:num_choices][:len_choice]
                        )
                        # 1d on max len context
                        len_context = gathered_len_context[i]
                        batch_tokenized_contexts_processed.append(gathered_contexts[i][:len_context])

                # Process the gathered results
                for i in range(len(batch_logits_sums)):
                    max_equals_doc = batch_max_equals[i]
                    logits_sum_doc = batch_logits_sums[i]
                    tokenized_contexts_batch = batch_tokenized_contexts_processed[i]
                    tokenized_continuations_batch = batch_tokenized_continuations_processed[i]
                    answer = ModelResponse(
                        argmax_logits_eq_gold=[max_equal.cpu().item() for max_equal in max_equals_doc],
                        logprobs=[sum.cpu().item() for sum in logits_sum_doc],
                        input_tokens=tokenized_contexts_batch,
                        output_tokens=tokenized_continuations_batch,
                    )
                    all_responses.append(answer)

        return dataset.get_original_order(all_responses)

    def prepare_batch_logprob(
        self,
        tokenized_contexts: list[list[list[int]]],
        tokenized_continuations: list[list[list[int]]],
        max_context: Optional[int] = None,
    ):
        """Tokenize a batch of inputs and return also the length, truncations and padding.
        This step is done manually since we tokenize log probability inputs together with their continuation,
        to manage possible extra spaces added at the start by tokenizers, see tok_encode_pair.
        """
        inputs = []
        # we used to remove the last token of continuation, but it's not needed
        for tokenized_context, tokenized_continuation in zip(tokenized_contexts, tokenized_continuations):
            inputs.extend(
                [context + continuation for context, continuation in zip(tokenized_context, tokenized_continuation)]
            )

        input_tokens = []
        attention_masks = []
        input_lengths = []
        truncated = []
        padded = []

        if max_context is None:
            max_context = self.max_length

        # First, find the longest sequence length in the batch
        max_sequence_length = max(len(seq) for seq in inputs)
        # If max_sequence_length is longer than max_context, we need to truncate
        effective_max_length = min(max_sequence_length, max_context)

        # Each sample is concatenated and cut to length or padded to max_length
        for orig_tokens in inputs:
            # Calculate truncation
            truncated.append(max(len(orig_tokens) - effective_max_length, 0))

            # Truncate from the left if needed to fit in the model's context
            tokens = torch.tensor((orig_tokens)[-effective_max_length:], dtype=torch.long).to(self.device)
            sequence_len = tokens.shape[0]

            # Calculate padding needed to reach effective_max_length
            padding_needed = effective_max_length - sequence_len
            padded.append(padding_needed)

            # Right padding to reach effective_max_length
            tokens = F.pad(tokens, (0, padding_needed), value=self.tokenizer.pad_token_id)

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = torch.ones_like(tokens, dtype=torch.bool)
            mask[sequence_len:] = False
            attention_masks.append(mask)

            input_tokens.append(tokens.unsqueeze(0))  # [1, effective_max_length]
            input_lengths.append(sequence_len)

        batched_inputs = torch.cat(input_tokens, dim=0)  # [batch, effective_max_length]
        attention_masks = torch.cat(attention_masks, dim=0)  # [batch, effective_max_length]

        return Batch(
            input_ids=batched_inputs,
            input_mask=attention_masks,
            input_lengths=input_lengths,
            truncated=truncated,
            padded=padded,
        )

    def pad_and_gather(
        self, output_tensor: torch.Tensor, drop_last_samples: bool = True, num_samples: int = None
    ) -> torch.Tensor:
        """Pads the `output_tensor` to the maximum length and gathers the lengths across processes.

        Args:
            output_tensor (torch.Tensor): The output tensor to be padded.
            drop_last_samples (bool, optional): Whether to drop the last samples during gathering.
                Last samples are dropped when the number of samples is not divisible by the number of processes.
                Defaults to True.
            num_samples (int, optional): Number of samples per batch item.

        Returns:
            torch.Tensor: The padded output tensor and the gathered length tensor.
        """
        # Create a tensor of size batch_size, [output_length] * batch_size, for each process
        # output_tensor can be of size: batch_size * num_samples * length_item or just batch_size * length_item
        length_tensor = torch.tensor([output_tensor.shape[-1]] * output_tensor.shape[0], device=self.device)
        if self.accelerator is not None:
            # Gather all the lengths, we end up with a tensor of size num_processes [output_length_1, output_length_2, ...]
            length_tensor = self.accelerator.gather(length_tensor)
        # We pad the output_tensor to the max length
        max_length = length_tensor.max().item()
        padding = (
            (0, max_length - output_tensor.shape[-1], 0, 0, 0, 0)
            if num_samples is not None
            else (0, max_length - output_tensor.shape[-1], 0, 0)
        )
        output_tensor = F.pad(output_tensor, padding, value=self.tokenizer.pad_token_id)
        if self.accelerator:
            if drop_last_samples:
                output_tensor = self.accelerator.gather_for_metrics(output_tensor)
            else:
                output_tensor = self.accelerator.gather(output_tensor)
        return output_tensor, length_tensor

    def tok_decode(self, tokens: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=self.skip_special_tokens)


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch: Batch = None,
        input_ids_shape: Tuple[int, int] = None,
    ):
        if batch is not None:
            initial_decoder_input_length = batch.input_ids.shape[1]
            batch_size = batch.input_ids.shape[0]
        else:
            initial_decoder_input_length = input_ids_shape[1]
            batch_size = input_ids_shape[0]

        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: list[str],
    batch: Batch,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[MultiTokenEOSCriteria(sequence, tokenizer, batch) for sequence in stop_sequences],
        ]
    )

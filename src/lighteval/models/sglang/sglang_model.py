# MIT License

# Copyright (c) 2024 The SGLang Team

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

import gc
import logging
from typing import Optional

import torch
from pydantic import PositiveFloat, PositiveInt
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig, _simplify_name
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_sglang_available


logger = logging.getLogger(__name__)

if is_sglang_available():
    from sglang import Engine
    from sglang.srt.hf_transformers_utils import get_tokenizer

    logging.getLogger("sglang").propagate = True
    logging.getLogger("sglang").handlers.clear()
else:
    Engine = None
    get_tokenizer = None


class SGLangModelConfig(ModelConfig):
    """
    Configuration class for SGLang inference engine.

    This configuration is used to load and configure models using the SGLang inference engine,
    which provides high-performance inference.

    sglang doc: https://docs.sglang.ai/index.html#

    Attributes:
        model_name (str):
            HuggingFace Hub model ID or path to the model to load.
        load_format (str):
            The format of the model weights to load. choices: auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral, runai_streamer.
        dtype (str):
            Data type for model weights. Defaults to "auto". Options: "auto", "float16", "bfloat16", "float32".
        tp_size (PositiveInt):
            Number of GPUs to use for tensor parallelism. Defaults to 1.
        dp_size (PositiveInt):
            Number of GPUs to use for data parallelism. Defaults to 1.
        context_length (PositiveInt | None):
            Maximum context length for the model.
        random_seed (PositiveInt | None):
            Random seed for reproducibility. Defaults to 1234.
        trust_remote_code (bool):
            Whether to trust remote code when loading models. Defaults to False.
        use_chat_template (bool):
            Whether to use chat templates for conversation-style prompts. Defaults to False.
        device (str):
            Device to load the model on. Defaults to "cuda".
        skip_tokenizer_init (bool):
            Whether to skip tokenizer initialization. Defaults to False.
        kv_cache_dtype (str):
            Data type for key-value cache. Defaults to "auto".
        add_special_tokens (bool):
            Whether to add special tokens during tokenization. Defaults to True.
        pairwise_tokenization (bool):
            Whether to tokenize context and continuation separately for loglikelihood evals. Defaults to False.
        sampling_backend (str | None):
            Sampling backend to use. If None, uses default.
        attention_backend (str | None):
            Attention backend to use. If None, uses default.
        mem_fraction_static (PositiveFloat):
            Fraction of GPU memory to use for static allocation. Defaults to 0.8.
        chunked_prefill_size (PositiveInt):
            Size of chunks for prefill operations. Defaults to 4096.

    Example:
        ```python
        config = SGLangModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            tp_size=2,
            context_length=8192,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    load_format: str = "auto"
    dtype: str = "auto"
    tp_size: PositiveInt = 1  # how many GPUs to use for tensor parallelism
    dp_size: PositiveInt = 1  # how many GPUs to use for data parallelism
    context_length: PositiveInt | None = None
    random_seed: PositiveInt | None = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    device: str = "cuda"
    skip_tokenizer_init: bool = False
    kv_cache_dtype: str = "auto"
    add_special_tokens: bool = True
    pairwise_tokenization: bool = False
    sampling_backend: str | None = None
    attention_backend: str | None = None
    mem_fraction_static: PositiveFloat = 0.8
    chunked_prefill_size: PositiveInt = 4096


class SGLangModel(LightevalModel):
    def __init__(
        self,
        config: SGLangModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.use_chat_template
        self.data_parallel_size = config.dp_size
        self.tensor_parallel_size = config.tp_size
        self._add_special_tokens = config.add_special_tokens
        self._tokenizer = self._create_auto_tokenizer(config)
        self._max_length = config.context_length if config.context_length is not None else None
        self.model = self._create_auto_model(config)
        self.model_name = _simplify_name(config.model_name)
        self.model_sha = ""  # config.get_model_sha()
        self.precision = config.dtype
        self.sampling_params = config.generation_parameters.to_sglang_dict()
        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
        self.sampling_backend = config.sampling_backend
        self.attention_backend = config.attention_backend
        self.pairwise_tokenization = config.pairwise_tokenization
        self.prompt_manager = PromptManager(self.use_chat_template, self.tokenizer, config.system_prompt)

    @property
    def tokenizer(self):
        return self._tokenizer

    def cleanup(self):
        if self.model is not None:
            self.model.shutdown()

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def _create_auto_model(self, config: SGLangModelConfig) -> Optional[Engine]:
        self.model_args = {
            "model_path": config.model_name,
            "trust_remote_code": config.trust_remote_code,
            "dtype": config.dtype,
            "device": "cuda",
            "random_seed": config.random_seed,
            "load_format": config.load_format,
            "context_length": self._max_length,
            "dp_size": config.dp_size,
            "tp_size": config.tp_size,
            "sampling_backend": config.sampling_backend,
            "attention_backend": config.attention_backend,
            "mem_fraction_static": config.mem_fraction_static,
            "schedule_policy": "fcfs",
            "chunked_prefill_size": config.chunked_prefill_size,
            "disable_radix_cache": True,
        }
        model = Engine(**self.model_args)

        if self._max_length is None:
            self._max_length = 8192

        return model

    def _create_auto_tokenizer(self, config: SGLangModelConfig):
        tokenizer = get_tokenizer(
            config.model_name,
            tokenizer_mode="auto",
            trust_remote_code=config.trust_remote_code,
            tokenizer_revision="main",
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,
        ):
            if self.use_chat_template:
                stop_tokens = []
            else:
                stop_tokens = split[0].stop_sequences

            max_new_tokens = split[0].generation_size  # could be none
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
                    inputs = [input[-context_size:] for input in inputs]
            else:
                if context_size > self.max_length:
                    logger.warning(
                        f"{context_size=} which is greater than {self.max_length=}. Truncating context to {self.max_length} tokens."
                    )
                    context_size = self.max_length
                    inputs = [input[-context_size:] for input in inputs]

            sglang_outputs = self._generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                num_samples=num_samples,
            )

            for input_token_ids, sglang_output in zip(inputs, sglang_outputs):
                meta_info = sglang_output["meta_info"]
                output_token_logprobs = meta_info["output_token_logprobs"]
                output_token_ids = [output[1] for output in output_token_logprobs]
                logprobs = [output[0] for output in output_token_logprobs]
                result = [sglang_output["text"]]
                cur_response = ModelResponse(
                    text=result,
                    logprobs=logprobs,
                    output_tokens=list(output_token_ids),
                    input_tokens=input_token_ids,
                )
                results.append(cur_response)
        return dataset.get_original_order(results)

    def _generate(
        self,
        inputs: list[list[int]],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        num_samples: int = 1,
        generate: bool = True,
    ) -> list:
        """Contains the actual logic of the generation."""

        logprob_start_len = None
        top_logprobs_num = None
        if generate:
            self.sampling_params["max_new_tokens"] = max_new_tokens
            self.sampling_params["stop"] = stop_tokens
            self.sampling_params["n"] = num_samples
            if num_samples > 1 and self.sampling_params["temperature"] == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )
        else:
            self.sampling_params["max_new_tokens"] = 1
            self.sampling_params["temperature"] = 0
            logprob_start_len = 0
            top_logprobs_num = 1

        outputs = self.model.generate(
            input_ids=inputs,
            sampling_params=self.sampling_params,
            return_logprob=True,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
        )
        return outputs

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        return self._loglikelihood_tokens(docs)

    def _loglikelihood_tokens(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=1)
        res = []

        for split in tqdm(dataset.splits_iterator(), disable=False):
            contexts = [self.prompt_manager.prepare_prompt(doc) for doc in split]
            # the last token is an eos token, so we don't need to add it
            # Left truncate the inputs to the maximum length
            inputs = []
            tokenized_continuations_batch = []
            tokenized_contexts_batch = []

            for context, doc in zip(contexts, dataset):
                tokenized_contexts, tokenized_continuations = self.tok_encode_pair(context, doc.choices, pairwise=True)
                for tokenized_context, tokenized_continuation in zip(tokenized_contexts, tokenized_continuations):
                    inputs.append(tokenized_context + tokenized_continuation)
                    tokenized_continuations_batch.append(tokenized_continuation)
                    tokenized_contexts_batch.append(tokenized_context)

            inputs = [input[-self.max_length :] for input in inputs]
            outputs = self._generate(inputs, generate=False)

            flat_index = 0
            for doc in dataset:
                # all the element generated from one doc (one element per choice)
                outputs_doc: list[dict] = outputs[flat_index : flat_index + len(doc.choices)]
                tokenized_continuations_doc: list[list[int]] = tokenized_continuations_batch[
                    flat_index : flat_index + len(doc.choices)
                ]
                tokenized_contexts_doc: list[list[int]] = tokenized_contexts_batch[
                    flat_index : flat_index + len(doc.choices)
                ]
                logprobs_doc = []
                argmax_doc = []
                output_tokens_doc = []
                input_tokens_doc = []

                for output, context, continuation in zip(
                    outputs_doc, tokenized_contexts_doc, tokenized_continuations_doc
                ):
                    meta_info = output["meta_info"]

                    input_top_logprobs = meta_info["input_top_logprobs"][::-1]
                    input_token_logprobs = meta_info["input_token_logprobs"][::-1]
                    input_top_logprobs = input_top_logprobs[: len(continuation)]
                    logprobs = input_token_logprobs[: len(continuation)]
                    bool_score = all(top[0][1] == input[1] for top, input in zip(input_top_logprobs, logprobs))
                    logprobs = [logprob[0] for logprob in logprobs]
                    logprobs_doc.append(logprobs)
                    argmax_doc.append(bool_score)
                    output_tokens_doc.append(output["text"])
                    input_tokens_doc.append(context + continuation)

                answer = ModelResponse(
                    input_tokens=input_tokens_doc,
                    output_tokens=output_tokens_doc,
                    logprobs=logprobs_doc,
                    argmax_logits_eq_gold=argmax_doc,
                )
                res.append(answer)
        return dataset.get_original_order(res)

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError()

    def loglikelihood_single_token(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError()

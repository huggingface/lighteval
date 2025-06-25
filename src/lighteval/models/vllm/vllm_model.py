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

import asyncio
import gc
import itertools
import logging
import os
from typing import Coroutine, Optional

import torch
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig, _simplify_name
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_vllm_available


logger = logging.getLogger(__name__)


if is_vllm_available():
    import ray
    from more_itertools import distribute
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.transformers_utils.tokenizer import get_tokenizer
    from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM

    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").handlers.clear()

    logging.getLogger("ray").propagate = True
    logging.getLogger("ray").handlers.clear()
else:
    from unittest.mock import Mock

    LLM = SamplingParams = get_tokenizer = ray = distribute = destroy_distributed_environment = (
        destroy_model_parallel
    ) = Mock()
    AsyncLLM = AsyncEngineArgs = RequestOutput = Mock()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class VLLMModelConfig(ModelConfig):
    """
    Configuration class for VLLM inference engine.

    This configuration is used to load and configure models using the VLLM inference engine,
    which provides high-performance inference for large language models with features like
    PagedAttention, continuous batching, and efficient memory management.

    vllm doc: https://docs.vllm.ai/en/v0.7.1/serving/engine_args.html

    Attributes:
        model_name (str):
            HuggingFace Hub model ID or path to the model to load.
        revision (str):
            Git revision of the model. Defaults to "main".
        dtype (str):
            Data type for model weights. Defaults to "bfloat16". Options: "float16", "bfloat16", "float32".
        tensor_parallel_size (PositiveInt):
            Number of GPUs to use for tensor parallelism. Defaults to 1.
        data_parallel_size (PositiveInt):
            Number of GPUs to use for data parallelism. Defaults to 1.
        pipeline_parallel_size (PositiveInt):
            Number of GPUs to use for pipeline parallelism. Defaults to 1.
        gpu_memory_utilization (NonNegativeFloat):
            Fraction of GPU memory to use. Lower this if running out of memory. Defaults to 0.9.
        max_model_length (PositiveInt | None):
            Maximum sequence length for the model. If None, automatically inferred.
            Reduce this if encountering OOM issues (4096 is usually sufficient).
        quantization (str | None):
            Quantization method.
        load_format (str | None):
            The format of the model weights to load. choices: auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral, runai_streamer.
        swap_space (PositiveInt):
            CPU swap space size in GiB per GPU. Defaults to 4.
        seed (NonNegativeInt):
            Random seed for reproducibility. Defaults to 1234.
        trust_remote_code (bool):
            Whether to trust remote code when loading models. Defaults to False.
        add_special_tokens (bool):
            Whether to add special tokens during tokenization. Defaults to True.
        multichoice_continuations_start_space (bool):
            Whether to add a space before multiple choice continuations. Defaults to True.
        pairwise_tokenization (bool):
            Whether to tokenize context and continuation separately for loglikelihood evals. Defaults to False.
        max_num_seqs (PositiveInt):
            Maximum number of sequences per iteration. Controls batch size at prefill stage. Defaults to 128.
        max_num_batched_tokens (PositiveInt):
            Maximum number of tokens per batch. Defaults to 2048.
        subfolder (str | None):
            Subfolder within the model repository. Defaults to None.
        use_chat_template (bool):
            Whether to use chat templates for conversation-style prompts. Defaults to False.
        is_async (bool):
            Whether to use the async version of VLLM. Defaults to False.

    Example:
        ```python
        config = VLLMModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_length=4096,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    revision: str = "main"  # revision of the model
    dtype: str = "bfloat16"
    tensor_parallel_size: PositiveInt = 1  # how many GPUs to use for tensor parallelism
    data_parallel_size: PositiveInt = 1  # how many GPUs to use for data parallelism
    pipeline_parallel_size: PositiveInt = 1  # how many GPUs to use for pipeline parallelism
    gpu_memory_utilization: NonNegativeFloat = 0.9  # lower this if you are running out of memory
    max_model_length: PositiveInt | None = (
        None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    )
    quantization: str | None = None
    load_format: str | None = None
    swap_space: PositiveInt = 4  # CPU swap space size (GiB) per GPU.
    seed: NonNegativeInt = 1234
    trust_remote_code: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = (
        True  # whether to add a space at the start of each continuation in multichoice generation
    )
    pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.
    max_num_seqs: PositiveInt = 128  # maximum number of sequences per iteration; This variable and `max_num_batched_tokens` effectively control the batch size at prefill stage. See https://github.com/vllm-project/vllm/issues/2492 for detailed explaination.
    max_num_batched_tokens: PositiveInt = 2048  # maximum number of tokens per batch
    subfolder: str | None = None
    use_chat_template: bool = False
    is_async: bool = False  # Whether to use the async version or sync version of the model


class VLLMModel(LightevalModel):
    def __init__(
        self,
        config: VLLMModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.use_chat_template
        self.data_parallel_size = config.data_parallel_size
        self.tensor_parallel_size = config.tensor_parallel_size
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config)

        self._max_length = config.max_model_length if config.max_model_length is not None else None

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = self._create_auto_model(config)

        # self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        self.model_name = _simplify_name(config.model_name)
        self.model_sha = ""
        self.precision = config.dtype

        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
        self.pairwise_tokenization = config.pairwise_tokenization

        self.prompt_manager = PromptManager(self.use_chat_template, self.tokenizer, config.system_prompt)

    @property
    def tokenizer(self):
        return self._tokenizer

    def cleanup(self):
        destroy_model_parallel()
        if self.model is not None:
            del self.model
        gc.collect()
        ray.shutdown()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def _create_auto_model(self, config: VLLMModelConfig) -> Optional[LLM]:
        """
        Creates an instance of the pretrained HF model.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model.
            subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
            max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
            device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
            torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
            quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
        }

        if config.quantization is not None:
            self.model_args["quantization"] = config.quantization
        if config.load_format is not None:
            self.model_args["load_format"] = config.load_format

        if config.data_parallel_size > 1:
            self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"
            return None

        model = LLM(**self.model_args)

        # If the max_length can't get extracted from the config, it will be inferred from the model
        # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
        # config and tk config, like mistralai/Mistral-7B-v0.1
        if self._max_length is None:
            self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

        return model

    def _create_auto_tokenizer(self, config: VLLMModelConfig):
        tokenizer = get_tokenizer(
            config.model_name,
            tokenizer_mode="auto",
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
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
            disable=False,  # self.disable_tqdm,
        ):
            # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
            if self.use_chat_template:
                stop_tokens = []
            else:
                # NOTE: we are assuming all items in a batch behave similarly (same
                # stop_tokens and max_tokens genrated) which is not necessarily
                # the case! Because of that we only use batch size of 1
                stop_tokens = split[0].stop_sequences or []

            max_new_tokens = self._config.generation_parameters.max_new_tokens or split[0].generation_size
            num_samples = split[0].num_samples

            context = [self.prompt_manager.prepare_prompt(doc) for doc in split]
            tokenized = self.tokenizer(context, add_special_tokens=self.add_special_tokens)

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

            vllm_outputs = self._generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                returns_logits=False,
                num_samples=num_samples,
            )

            for i, vllm_output in enumerate(vllm_outputs):
                output_token_ids = [outputs.token_ids for outputs in vllm_output.outputs]
                result = [output.text for output in vllm_output.outputs]
                input_token_ids = vllm_output.prompt_token_ids

                cur_response = ModelResponse(
                    input=context[i],
                    text=result,
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
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
    ) -> list:
        """Contains the actual logic of the generation."""
        sampling_params = SamplingParams(**self._config.generation_parameters.to_vllm_dict())

        if generate:
            sampling_params.n = num_samples
            sampling_params.max_tokens = max_new_tokens
            sampling_params.stop = stop_tokens
            sampling_params.logprobs = 1 if returns_logits else 0
            if num_samples > 1 and sampling_params.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )
        else:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False

        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            # Hynek: With the newest vllm, it actually breaks when tensor_parallel_size == 1 and num_gpus not set,
            # as VLLM complains about no GPUs available.
            @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
            def run_inference_one_model(model_args: dict, sampling_params: SamplingParams, requests):
                llm = LLM(**model_args)
                return llm.generate(prompt_token_ids=requests, sampling_params=sampling_params)

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, inputs)]
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            outputs = [
                x
                for x in itertools.chain.from_iterable(itertools.zip_longest(*[list(x) for x in results]))
                if x is not None
            ]
        else:
            outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
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

        for split in tqdm(dataset.splits_iterator()):
            contexts = [self.prompt_manager.prepare_prompt(doc) for doc in split]

            inputs = []
            tokenized_continuations_batch = []
            tokenized_contexts_batch = []

            for context, doc in zip(contexts, split):
                tokenized_contexts, tokenized_continuations = self.tok_encode_pair(
                    context, doc.choices, pairwise=self.pairwise_tokenization
                )
                for tokenized_context, tokenized_continuation in zip(tokenized_contexts, tokenized_continuations):
                    inputs.append(tokenized_context + tokenized_continuation)
                    tokenized_continuations_batch.append(tokenized_continuation)
                    tokenized_contexts_batch.append(tokenized_context)

            # Left truncate the inputs to the maximum length
            inputs = [input[-self.max_length :] for input in inputs]
            outputs = self._generate(inputs, generate=False)

            flat_index = 0
            for i, doc in enumerate(split):
                outputs_doc = outputs[flat_index : flat_index + len(doc.choices)]
                tokenized_continuations_doc = tokenized_continuations_batch[flat_index : flat_index + len(doc.choices)]
                tokenized_contexts_doc = tokenized_contexts_batch[flat_index : flat_index + len(doc.choices)]
                logprobs_doc = []
                argmax_doc = []
                output_tokens_doc = []
                input_tokens_doc = []

                for output, context, continuation in zip(
                    outputs_doc, tokenized_contexts_doc, tokenized_continuations_doc
                ):
                    continuation_logprobs = []
                    for token, logprobs in zip(continuation[::-1], output.prompt_logprobs[::-1]):
                        continuation_logprobs.append(logprobs[token])

                    bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                    continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
                    continuation_logprobs = sum(continuation_logprobs)
                    logprobs_doc.append(continuation_logprobs)
                    argmax_doc.append(bool_score)
                    output_tokens_doc.append(continuation)
                    input_tokens_doc.append(context)

                answer = ModelResponse(
                    input=contexts[i],
                    input_tokens=input_tokens_doc,
                    output_tokens=output_tokens_doc,
                    logprobs=logprobs_doc,
                    argmax_logits_eq_gold=argmax_doc,
                )
                res.append(answer)
                flat_index += len(doc.choices)

        return dataset.get_original_order(res)

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError()


class AsyncVLLMModel(VLLMModel):
    """VLLM models which deploy async natively (no ray). Supports DP and PP/TP but not batch size > 1"""

    DATASET_SPLITS = 1
    is_async = True

    def cleanup(self):
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    def _create_auto_model(self, config: VLLMModelConfig):
        """
        Creates an instance of the async vllm model loaded from HF. Requires using the v1 of VLLM.

        Returns:
            AsyncLLM: The created async VLLM instance
        """
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "data_parallel_size": config.data_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
            "enforce_eager": True,
        }

        if config.data_parallel_size > 1:
            self._batch_size = "auto"

        model = AsyncLLM.from_engine_args(AsyncEngineArgs(**self.model_args))

        # If the max_length can't get extracted from the config, it will be inferred from the model
        if self._max_length is None:
            self._max_length = model.model_config.max_seq_len_to_capture

        return model

    async def _async_one_item(
        self,
        index: int,
        doc: Doc,
        generative: bool,
    ) -> Coroutine[None, list, str]:
        """Contains the actual logic of the generation."""
        sampling_params = SamplingParams(**self._config.generation_parameters.to_vllm_dict())

        if not generative:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False
            prompt = self.prompt_manager.prepare_prompt(doc) + doc.choice
            index_str = f"logprob_{index}"
        else:
            sampling_params.n = doc.num_samples
            if sampling_params.n > 1:
                # Todo clementine: investigate more
                logger.warning(
                    "Careful, there can be unexpected behavior when using sampling evals with the async vllm model"
                )
            sampling_params.max_tokens = self._config.generation_parameters.max_new_tokens or doc.generation_size
            sampling_params.stop = [] if self.use_chat_template else doc.stop_sequences
            sampling_params.logprobs = int(doc.use_logits)
            prompt = self.prompt_manager.prepare_prompt(doc)
            index_str = f"generative_{index}"

        generator = self.model.generate(request_id=index_str, prompt=prompt, sampling_params=sampling_params)
        try:
            while output := await anext(generator):
                continue
        except StopAsyncIteration:
            pass

        return output

    async def _async_batch(self, docs: list[Doc], generative: bool) -> list:
        processed_requests = [
            self._async_one_item(index=index, doc=doc, generative=generative) for index, doc in enumerate(docs)
        ]
        results = await asyncio.gather(*processed_requests)
        return results

    async def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        results = []

        responses = await self._async_batch(docs=docs, generative=True)

        for response in responses:
            output_token_ids = [outputs.token_ids for outputs in response.outputs]
            full_logprobs = [output.logprobs for output in response.outputs] or []
            logprobs = [logprob[token_id].logprob for token_id, logprob in zip(output_token_ids[0], full_logprobs[0])]
            result = [output.text for output in response.outputs]
            input_token_ids = response.prompt_token_ids

            cur_response = ModelResponse(
                text=result,
                logprobs=logprobs,
                output_tokens=list(output_token_ids),
                input_tokens=input_token_ids,
            )
            results.append(cur_response)

        return results

    async def loglikelihood(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met and
        stores the logprobs.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.

        Returns:
            list[LoglikelihoodResponse]: list of generated responses.
        """
        results = []

        responses = await self._async_batch(docs=docs, generative=False)

        for response, input in zip(responses, docs):
            continuation_logprobs = []
            for token, logprobs in zip(input.tokenized_continuation[::-1], response.prompt_logprobs[::-1]):
                continuation_logprobs.append(logprobs[token])
            bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
            continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
            answer = ModelResponse(
                input_tokens=input.tokenized_context + input.tokenized_continuation,
                output_tokens=input.tokenized_continuation,
                logprobs=sum(continuation_logprobs),
                argmax_logits_eq_gold=bool_score,
            )
            results.append(answer)

        return results

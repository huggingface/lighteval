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

import gc
import itertools
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
)
from lighteval.models.utils import _get_dtype, _simplify_name
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
)
from lighteval.utils.imports import is_vllm_available
from lighteval.utils.utils import EnvConfig, as_list


logger = logging.getLogger(__name__)


if is_vllm_available():
    import ray
    from more_itertools import distribute
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from vllm.transformers_utils.tokenizer import get_tokenizer

    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").handlers.clear()

    logging.getLogger("ray").propagate = True
    logging.getLogger("ray").handlers.clear()
else:
    LLM = None
    SamplingParams = None
    get_tokenizer = None
    ray = None
    distribute = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


@dataclass
class VLLMModelConfig:
    pretrained: str
    gpu_memory_utilization: float = 0.9  # lower this if you are running out of memory
    revision: str = "main"  # revision of the model
    dtype: str | None = None
    tensor_parallel_size: int = 1  # how many GPUs to use for tensor parallelism
    pipeline_parallel_size: int = 1  # how many GPUs to use for pipeline parallelism
    data_parallel_size: int = 1  # how many GPUs to use for data parallelism
    max_model_length: int | None = None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    swap_space: int = 4  # CPU swap space size (GiB) per GPU.
    seed: int = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = (
        True  # whether to add a space at the start of each continuation in multichoice generation
    )
    pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.
    generation_parameters: GenerationParameters = None  # sampling parameters to use for generation

    subfolder: Optional[str] = None

    def __post_init__(self):
        if not self.generation_parameters:
            self.generation_parameters = GenerationParameters()


class VLLMModel(LightevalModel):
    def __init__(
        self,
        config: VLLMModelConfig,
        env_config: EnvConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.use_chat_template
        self.data_parallel_size = int(config.data_parallel_size)
        self.tensor_parallel_size = int(config.tensor_parallel_size)

        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config, env_config)

        self._max_length = int(config.max_model_length) if config.max_model_length is not None else None

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = self._create_auto_model(config, env_config)

        # self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = ""  # config.get_model_sha()
        self.precision = _get_dtype(config.dtype, config=self._config)

        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
        self.sampling_params = SamplingParams(**config.generation_parameters.to_vllm_dict())
        self.pairwise_tokenization = config.pairwise_tokenization

    @property
    def tokenizer(self):
        return self._tokenizer

    def cleanup(self):
        destroy_model_parallel()
        if self.model is not None:
            del self.model.llm_engine.model_executor.driver_worker
        self.model = None
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

    def _create_auto_model(self, config: VLLMModelConfig, env_config: EnvConfig) -> Optional[LLM]:
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
            "model": config.pretrained,
            "gpu_memory_utilization": float(config.gpu_memory_utilization),
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": int(config.tensor_parallel_size),
            "pipeline_parallel_size": int(config.pipeline_parallel_size),
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": 1234,
        }
        if int(config.data_parallel_size) > 1:
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

    def _create_auto_tokenizer(self, config: VLLMModelConfig, env_config: EnvConfig):
        tokenizer = get_tokenizer(
            config.pretrained,
            tokenizer_mode="auto",
            trust_remote_code=config.trust_remote_code,
            tokenizer_revision=config.revision,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for _ in tqdm(
            dataset.splits_start_end_iterator(),
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
                stop_tokens = dataset[0].stop_sequence

            max_new_tokens = dataset[0].generation_size  # could be none
            returns_logits = dataset[0].use_logits
            num_samples = dataset[0].num_samples

            context = [c.context for c in dataset]
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
                        f"{context_size + max_new_tokens=} which is greather than {self.max_length=}. Truncating context to {self.max_length - max_new_tokens} tokens."
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
                        f"{context_size=} which is greather than {self.max_length=}. Truncating context to {self.max_length} tokens."
                    )
                    context_size = self.max_length
                    inputs = [input[-context_size:] for input in inputs]

            vllm_outputs = self._generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                returns_logits=returns_logits,
                num_samples=num_samples,
            )

            for vllm_output in vllm_outputs:
                output_token_ids = [outputs.token_ids for outputs in vllm_output.outputs]
                logprobs = [output.logprobs for output in vllm_output.outputs] or []
                logprobs = [logprob[token_id].logprob for token_id, logprob in zip(output_token_ids[0], logprobs[0])]
                result = [output.text for output in vllm_output.outputs]
                input_token_ids = vllm_output.prompt_token_ids

                cur_response = GenerativeResponse(
                    result=result,
                    logits=logprobs,
                    generated_tokens=list(output_token_ids),
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
    ) -> list[GenerativeResponse]:
        """Contains the actual logic of the generation."""
        sampling_params = self.sampling_params.clone() or SamplingParams()
        if generate:
            sampling_params.n = num_samples
            sampling_params.max_tokens = (
                max_new_tokens if sampling_params.max_tokens is None else sampling_params.max_tokens
            )
            sampling_params.stop = stop_tokens
            sampling_params.logprobs = 1 if returns_logits else 0

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

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice, pairwise=self.pairwise_tokenization
                )
        return self._loglikelihood_tokens(requests, override_bs=override_bs)

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
        override_bs: int = -1,
        return_bool_score: bool = True,
        rolling: bool = False,
    ) -> list[LoglikelihoodResponse]:
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=1)
        res = []

        for _ in tqdm(dataset.splits_start_end_iterator()):
            # the last token is an eos token, so we don't need to add it
            inputs = [dataset[i].tokenized_context + dataset[i].tokenized_continuation for i in range(len(dataset))]
            # Left truncate the inputs to the maximum length
            inputs = [input[-self.max_length :] for input in inputs]
            outputs = self._generate(inputs, generate=False)

            for output, input in zip(outputs, dataset):
                continuation_logprobs = []
                for token, logprobs in zip(input.tokenized_continuation[::-1], output.prompt_logprobs[::-1]):
                    continuation_logprobs.append(logprobs[token])
                bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
                answer = LoglikelihoodResponse(
                    input_tokens=input.tokenized_context + input.tokenized_continuation,
                    generated_tokens=input.tokenized_continuation,
                    result=(sum(continuation_logprobs), bool_score if return_bool_score else None),
                )
                res.append(answer)

        return dataset.get_original_order(res)

    def loglikelihood_rolling():
        pass

    def loglikelihood_single_token():
        pass

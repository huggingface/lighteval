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

import itertools
import os
from typing import Optional

from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_config import VLLMModelConfig
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


if is_vllm_available():
    import ray
    from more_itertools import distribute
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.tokenizer import get_tokenizer
else:
    LLM = None
    SamplingParams = None
    get_tokenizer = None
    ray = None
    distribute = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class VLLMModel(LightevalModel):
    def __init__(
        self,
        config: VLLMModelConfig,
        env_config: EnvConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self._batch_size = config.batch_size
        self._max_length = self._init_max_length(config.max_model_length)
        self.use_chat_template = config.use_chat_template
        self.data_parallel_size = int(config.data_parallel_size)

        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config, env_config)

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = self._create_auto_model(config, env_config)

        # self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = ""  # config.get_model_sha()
        self.precision = _get_dtype(config.dtype, config=self._config)

        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)

    @property
    def tokenizer(self):
        return self._tokenizer

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
            "gpu_memory_utilization": float(0.8),
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": int(1),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "swap_space": 4,
            "seed": 1234,
        }
        if int(config.data_parallel_size) > 1:
            self.model_args["worker_use_ray"] = True
            self._batch_size = "auto"
            return None

        model = LLM(**self.model_args)
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

    def _init_max_length(self, max_length) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.

        Args:
            max_length (Optional[int]): The maximum length of the input sequence. If not provided, it will be determined
                based on the model's configuration or tokenizer's model_max_length attribute.

        Returns:
            int: Max length to use depending on the available args and config
        """
        if max_length is not None:
            return int(max_length)
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")

        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)

        # Default max sequence length setting for when no `max_length` is provided
        # or no max length config setting is found in the model or tokenizer.
        return 2048

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
            context_size = len(tokenized["input_ids"][0])
            if context_size > self.max_length:
                hlog_warn(
                    f"The context size of your batch ({context_size}) is bigger than the maximum context size allowed by the model ({self.max_length}) for a task in"
                    + str({dataset[0].task_name})
                    + ". This is likely to lead to some errors."  # noqa C401
                )
                # There will be truncation of at least one sample, maximum generation size will be one
                max_new_tokens = 1
            else:  # We can't allow generation of more than max_length
                if max_new_tokens is None:  # If generation size is not set, we go all the way
                    max_new_tokens = self.max_length - context_size
                else:
                    max_new_tokens = min(self.max_length - context_size, max_new_tokens)

            vllm_outputs = self._generate(
                inputs=tokenized["input_ids"],
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                returns_logits=returns_logits,
                num_samples=num_samples,
            )

            print(f"{len(vllm_outputs)} vllm_outputs")
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
        if generate:
            sampling_params = SamplingParams(
                n=num_samples, max_tokens=max_new_tokens, stop=stop_tokens, logprobs=1 if returns_logits else 0
            )
        else:
            sampling_params = SamplingParams(temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False)

        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            @ray.remote
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
                    request.context, request.choice
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
            inputs = [
                dataset[i].tokenized_context + dataset[i].tokenized_continuation[:-1] for i in range(len(dataset))
            ]
            outputs = self._generate(inputs, generate=False)

            for output, input in zip(outputs, dataset):
                continuation_logprobs = []
                for token, logprobs in zip(input.tokenized_continuation[-2::-1], output.prompt_logprobs[::-1]):
                    continuation_logprobs.append(logprobs[token])
                bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
                answer = LoglikelihoodResponse(
                    result=(sum(continuation_logprobs), bool_score if return_bool_score else None)
                )
                res.append(answer)

        return dataset.get_original_order(res)

    def loglikelihood_rolling():
        pass

    def loglikelihood_single_token():
        pass

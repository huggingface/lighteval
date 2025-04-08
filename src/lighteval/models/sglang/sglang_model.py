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
from lighteval.utils.imports import is_sglang_available
from lighteval.utils.utils import EnvConfig, as_list


logger = logging.getLogger(__name__)

if is_sglang_available():
    from sglang import Engine
    from sglang.srt.hf_transformers_utils import get_tokenizer

    logging.getLogger("sglang").propagate = True
    logging.getLogger("sglang").handlers.clear()
else:
    Engine = None
    get_tokenizer = None


@dataclass
class SGLangModelConfig:
    pretrained: str
    load_format: str = "auto"
    dtype: str = "auto"
    tp_size: int = 1  # how many GPUs to use for tensor parallelism
    dp_size: int = 1  # how many GPUs to use for data parallelism
    context_length: int | None = None
    random_seed: Optional[int] = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    device: str = "cuda"
    skip_tokenizer_init: bool = False
    kv_cache_dtype: str = "auto"
    add_special_tokens: bool = True
    pairwise_tokenization: bool = False
    sampling_backend: str | None = None
    attention_backend: str = None
    mem_fraction_static: float = 0.8
    chunked_prefill_size: int = 4096
    generation_parameters: GenerationParameters = None

    def __post_init__(self):
        if not self.generation_parameters:
            self.generation_parameters = GenerationParameters()


class SGLangModel(LightevalModel):
    def __init__(
        self,
        config: SGLangModelConfig,
        env_config: EnvConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.use_chat_template
        self.data_parallel_size = int(config.dp_size)
        self.tensor_parallel_size = int(config.tp_size)
        self._add_special_tokens = bool(config.add_special_tokens)
        self._tokenizer = self._create_auto_tokenizer(config, env_config)
        self._max_length = int(config.context_length) if config.context_length is not None else None
        self.model = self._create_auto_model(config, env_config)
        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = ""  # config.get_model_sha()
        self.precision = _get_dtype(config.dtype, config=self._config)
        self.sampling_params = config.generation_parameters.to_sglang_dict()
        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
        self.sampling_backend = config.sampling_backend
        self.attention_backend = config.attention_backend
        self.pairwise_tokenization = config.pairwise_tokenization

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

    def _create_auto_model(self, config: SGLangModelConfig, env_config: EnvConfig) -> Optional[Engine]:
        self.model_args = {
            "model_path": config.pretrained,
            "trust_remote_code": config.trust_remote_code,
            "dtype": config.dtype,
            "device": "cuda",
            "random_seed": config.random_seed,
            "load_format": config.load_format,
            "context_length": self._max_length,
            "dp_size": int(config.dp_size),
            "tp_size": int(config.tp_size),
            "sampling_backend": config.sampling_backend,
            "attention_backend": config.attention_backend,
            "mem_fraction_static": float(config.mem_fraction_static),
            "schedule_policy": "fcfs",
            "chunked_prefill_size": int(config.chunked_prefill_size),
            "disable_radix_cache": True,
        }
        model = Engine(**self.model_args)

        if self._max_length is None:
            self._max_length = 8192

        return model

    def _create_auto_tokenizer(self, config: SGLangModelConfig, env_config: EnvConfig):
        tokenizer = get_tokenizer(
            config.pretrained,
            tokenizer_mode="auto",
            trust_remote_code=config.trust_remote_code,
            tokenizer_revision="main",
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
            disable=False,
        ):
            if self.use_chat_template:
                stop_tokens = []
            else:
                stop_tokens = dataset[0].stop_sequence

            max_new_tokens = dataset[0].generation_size  # could be none
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
        num_samples: int = 1,
        generate: bool = True,
    ) -> list[GenerativeResponse]:
        """Contains the actual logic of the generation."""

        logprob_start_len = None
        top_logprobs_num = None
        if generate:
            self.sampling_params["max_new_tokens"] = max_new_tokens
            self.sampling_params["stop"] = stop_tokens
            self.sampling_params["n"] = num_samples
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

        for _ in tqdm(dataset.splits_start_end_iterator(), disable=False):
            # the last token is an eos token, so we don't need to add it
            inputs = [dataset[i].tokenized_context + dataset[i].tokenized_continuation for i in range(len(dataset))]
            # Left truncate the inputs to the maximum length
            inputs = [input[-self.max_length :] for input in inputs]
            outputs = self._generate(inputs, generate=False)

            for output, input in zip(outputs, dataset):
                continuation_logprobs = []
                meta_info = output["meta_info"]
                input_token_logprobs = meta_info["input_token_logprobs"][::-1]
                input_top_logprobs = meta_info["input_top_logprobs"][::-1]
                input_top_logprobs = input_top_logprobs[: len(input.tokenized_continuation)]
                continuation_logprobs.append(input_token_logprobs[: len(input.tokenized_continuation)])
                bool_score = all(
                    top[0][1] == input[1] for top, input in zip(input_top_logprobs, continuation_logprobs[0])
                )
                answer = LoglikelihoodResponse(
                    input_tokens=input.tokenized_context + input.tokenized_continuation,
                    generated_tokens=input.tokenized_continuation,
                    result=(sum(item[0] for item in continuation_logprobs[0]), bool_score),
                )
                res.append(answer)
        return dataset.get_original_order(res)

    def loglikelihood_rolling():
        pass

    def loglikelihood_single_token():
        pass

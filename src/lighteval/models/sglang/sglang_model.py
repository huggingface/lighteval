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
from lighteval.utils.imports import is_sglang_available
from lighteval.utils.utils import EnvConfig, as_list


logger = logging.getLogger(__name__)

from more_itertools import distribute
from sglang import Engine
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

logging.getLogger("sglang").propagate = True
logging.getLogger("sglang").handlers.clear()

## Jayon02: sglang with what dependency, ray? flashinfer?
# if is_sglang_available():
#     from more_itertools import distribute
#     from vllm import LLM, SamplingParams
#     from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
#     from vllm.transformers_utils.tokenizer import get_tokenizer
#
#     logging.getLogger("sglang").propagate = True
#     logging.getLogger("sglang").handlers.clear()
# else:
#     LLM = None
#     SamplingParams = None
#     get_tokenizer = None
#     distribute = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512

## change to all sglang config
@dataclass
class SGLANGModelConfig:
    pretrained: str #
    load_format: str = "auto"
    dtype: str = "auto" #
    tp_size: int = 1  # how many GPUs to use for tensor parallelism
    dp_size: int = 1  # how many GPUs to use for data parallelism
    context_length: int | None = None
    random_seed: Optional[int] = None
    trust_remote_code: bool = True #
    chat_template: Optional[str] = None
    device: str = "cuda"
    skip_tokenizer_init: bool = False
    kv_cache_dtype: str = "auto",
    add_special_tokens: bool = True,
    pipeline_parallel_size: int = 1  # how many GPUs to use for pipeline parallelism

class SGLANGModel(LightevalModel):
    def __init__(
        self,
        config: SGLANGModelConfig,
        env_config: EnvConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.chat_template is not None
        self.data_parallel_size = int(config.dp_size)
        self.tensor_parallel_size = int(config.tp_size)
        self._add_special_tokens = bool(config.add_special_tokens)
        self._tokenizer = self._create_auto_tokenizer(config, env_config)
        self._max_length = int(config.context_length) if config.context_length is not None else 256
        self.model = self._create_auto_model(config, env_config)
        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = ""  # config.get_model_sha()
        self.precision = _get_dtype(config.dtype, config=self._config)
        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
 
    @property
    def tokenizer(self):
        return self._tokenizer

    def cleanup(self):
        destroy_model_parallel()
        if self.model is not None:
            self.model.shutdown()
        self.model = None
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def _create_auto_model(self, config: SGLANGModelConfig, env_config: EnvConfig) -> Optional[Engine]:
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
        # TODO: double check
        self.model_args  = {
            "model_path": config.pretrained,
            "trust_remote_code": config.trust_remote_code,
            "dtype": config.dtype,
            "device": "cuda",
            "random_seed": config.random_seed,
            "load_format": config.load_format,
            "context_length": int(self._max_length) if self._max_length else None,
            "dp_size": int(config.dp_size),
            "tp_size": int(config.tp_size),
            "log_level": "info",
        }

        if config.dp_size > 1:
            pass

        model = Engine(**self.model_args)

        # TODO: double check
        # If the max_length can't get extracted from the config, it will be inferred from the model
        # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
        # config and tk config, like mistralai/Mistral-7B-v0.1
        # if self._max_length is None:
        #    self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

        return model

    def _create_auto_tokenizer(self, config: SGLANGModelConfig, env_config: EnvConfig):
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
                    inputs = [input[-context_size:] for input in inputs]
            else:
                if context_size > self.max_length:
                    logger.warning(
                        f"{context_size=} which is greather than {self.max_length=}. Truncating context to {self.max_length} tokens."
                    )
                    context_size = self.max_length
                    inputs = [input[-context_size:] for input in inputs]

            sglang_outputs = self._generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                num_samples=num_samples,
            )

            for i in range(len(sglang_outputs)):
                sglang_output = sglang_outputs[i]
                # print(sglang_output)
                # exit(0)
                meta_info = sglang_output["meta_info"]
                output_token_logprobs = meta_info["output_token_logprobs"]
                output_token_ids = [output[1] for output in output_token_logprobs]
                logprobs = [output[0] for output in output_token_logprobs]
                result = [sglang_output["text"]]
                input_token_ids = inputs[i]
        
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
        # TODO: double check without clone

        params = dict(
            top_p=1.0,
            top_k=-1,
            min_p=0,
            max_new_tokens=max_new_tokens,
            stop=stop_tokens,
            temperature=1.0,
            repetition_penalty=1.0,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            n=num_samples
        )

        if not generate:
            params.temperature = 0
            params.max_tokens = 1

        outputs = self.model.generate(
                input_ids=inputs,
                sampling_params=params,
                return_logprob=True,
            )

        return outputs

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        pass

    def loglikelihood_rolling():
        pass

    def loglikelihood_single_token():
        pass

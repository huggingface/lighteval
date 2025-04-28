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
import logging
import re
import time
from dataclasses import dataclass, replace
from typing import Coroutine, Dict, List, Optional, Union

import requests
import torch
from huggingface_hub import (
    AsyncInferenceClient,
    InferenceClient,
    InferenceEndpoint,
    InferenceEndpointError,
    InferenceEndpointTimeoutError,
    TextGenerationInputGenerateParameters,
    TextGenerationInputGrammarType,
    TextGenerationOutput,
    create_inference_endpoint,
    get_inference_endpoint,
)
from huggingface_hub.utils import HfHubHTTPError
from requests import ConnectionError
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse, LoglikelihoodSingleTokenResponse
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.utils import EnvConfig, as_list


logger = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_TIME_FOR_SPINUP = 3600

SORTED_INSTANCE_SIZES = [  # sorted by incremental overall RAM (to load models)
    # type, size
    ("nvidia-a10g", "x1"),
    ("nvidia-t4", "x4"),
    ("nvidia-a100", "x1"),
    ("nvidia-a10g", "x4"),
    ("nvidia-a100", "x2"),
    ("nvidia-a100", "x4"),
]


@dataclass
class ServerlessEndpointModelConfig:
    model_name: str
    add_special_tokens: bool = True
    generation_parameters: GenerationParameters = None

    def __post_init__(self):
        if not self.generation_parameters:
            self.generation_parameters = GenerationParameters()

    @classmethod
    def from_path(cls, path: str) -> "ServerlessEndpointModelConfig":
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f)["model"]
        return cls(**config["base_params"])


@dataclass
class InferenceEndpointModelConfig:
    endpoint_name: str = None
    model_name: str = None
    reuse_existing: bool = False
    accelerator: str = "gpu"
    model_dtype: str = None  # if empty, we use the default
    vendor: str = "aws"
    region: str = "us-east-1"  # this region has the most hardware options available
    instance_size: str = None  # if none, we autoscale
    instance_type: str = None  # if none, we autoscale
    framework: str = "pytorch"
    endpoint_type: str = "protected"
    add_special_tokens: bool = True
    revision: str = "main"
    namespace: str = None  # The namespace under which to launch the endpoint. Defaults to the current user's namespace
    image_url: str = None
    env_vars: dict = None
    generation_parameters: GenerationParameters = None

    def __post_init__(self):
        # xor operator, one is None but not the other
        if (self.instance_size is None) ^ (self.instance_type is None):
            raise ValueError(
                "When creating an inference endpoint, you need to specify explicitly both instance_type and instance_size, or none of them for autoscaling."
            )

        if not (self.endpoint_name is None) ^ int(self.model_name is None):
            raise ValueError("You need to set either endpoint_name or model_name (but not both).")

        if not self.generation_parameters:
            self.generation_parameters = GenerationParameters()

    @classmethod
    def from_path(cls, path: str) -> "InferenceEndpointModelConfig":
        """Load configuration for inference endpoint model from YAML file path.

        Args:
            path (`str`): Path of the model configuration YAML file.

        Returns:
            [`InferenceEndpointModelConfig`]: Configuration for inference endpoint model.
        """
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f)["model"]
        config["base_params"]["model_dtype"] = config["base_params"].pop("dtype", None)
        return cls(**config["base_params"], **config.get("instance", {}))

    def get_dtype_args(self) -> Dict[str, str]:
        if self.model_dtype is None:
            return {}
        model_dtype = self.model_dtype.lower()
        if model_dtype in ["awq", "eetq", "gptq"]:
            return {"QUANTIZE": model_dtype}
        if model_dtype == "8bit":
            return {"QUANTIZE": "bitsandbytes"}
        if model_dtype == "4bit":
            return {"QUANTIZE": "bitsandbytes-nf4"}
        if model_dtype in ["bfloat16", "float16"]:
            return {"DTYPE": model_dtype}
        return {}

    def get_custom_env_vars(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.env_vars.items()} if self.env_vars else {}


class InferenceEndpointModel(LightevalModel):
    """InferenceEndpointModels can be used both with the free inference client, or with inference
    endpoints, which will use text-generation-inference to deploy your model for the duration of the evaluation.
    """

    def __init__(  # noqa: C901
        self, config: Union[InferenceEndpointModelConfig, ServerlessEndpointModelConfig], env_config: EnvConfig
    ) -> None:
        self.reuse_existing = getattr(config, "reuse_existing", False)
        self._max_length = None
        self.endpoint = None
        self.model_name = None
        if isinstance(config, InferenceEndpointModelConfig):
            if config.instance_type and config.instance_size and config.vendor and config.region:
                vendor, region, instance_type, instance_size = (
                    config.vendor,
                    config.region,
                    config.instance_type,
                    config.instance_size,
                )
            else:
                try:
                    vendor, region, instance_type, instance_size = InferenceEndpointModel.get_suggested_model_config(
                        config.model_name
                    )
                except Exception:
                    vendor, region, instance_type, instance_size = (
                        "aws",
                        "us-east-1",
                        *InferenceEndpointModel.get_larger_hardware_suggestion(),
                    )

            must_scaleup_endpoint = False
            timer_start = time.time()
            # Endpoint names do not allow special characters
            endpoint_name = config.endpoint_name or re.sub(
                "[^a-zA-Z0-9-]", "-", config.model_name.lower() + "-lighteval"
            )
            # If no endpoint or endpoint not running, and we're below an hour
            while (self.endpoint is None or self.endpoint.status != "running") and (
                time.time() - timer_start < MAX_TIME_FOR_SPINUP
            ):
                try:
                    if self.endpoint is None:  # Endpoint does not exist yet locally
                        if not config.reuse_existing:  # New endpoint
                            logger.info("Creating endpoint.")
                            self.endpoint: InferenceEndpoint = create_inference_endpoint(
                                name=endpoint_name,
                                namespace=config.namespace,
                                repository=config.model_name,
                                revision=config.revision,
                                framework=config.framework,
                                task="text-generation",
                                accelerator=config.accelerator,
                                type=config.endpoint_type,
                                vendor=vendor,
                                region=region,
                                instance_size=instance_size,
                                instance_type=instance_type,
                                token=env_config.token,
                                custom_image={
                                    "health_route": "/health",
                                    "env": {
                                        # Documentation: https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/launcher
                                        "MAX_BATCH_PREFILL_TOKENS": "2048",
                                        "MAX_INPUT_LENGTH": "2047",
                                        "MAX_TOTAL_TOKENS": "2048",
                                        "MODEL_ID": "/repository",
                                        "HF_MODEL_TRUST_REMOTE_CODE": "true",
                                        **config.get_dtype_args(),
                                        **config.get_custom_env_vars(),
                                    },
                                    "url": (config.image_url or "ghcr.io/huggingface/text-generation-inference:3.0.1"),
                                },
                            )
                        else:  # Endpoint exists
                            logger.info("Reusing existing endpoint.")
                            self.endpoint = get_inference_endpoint(
                                name=endpoint_name, token=env_config.token, namespace=config.namespace
                            )

                    else:
                        # Endpoint exists locally but either failed (and most likely it must be scaled up)
                        if must_scaleup_endpoint:
                            logger.info("Rescaling existing endpoint.")
                            self.endpoint.update(instance_size=instance_size, instance_type=instance_type)
                            must_scaleup_endpoint = False
                        # or we got a connection error, in which case we do nothing and just wait at the next step

                    # Waits for the endpoint to be deployed - we could also check for the status in updating', 'pending', 'initializing'
                    logger.info("Trying to deploy your endpoint. Please wait for 10 min.")
                    self.endpoint.wait(timeout=600, refresh_every=60)  # We wait for 10 min
                except InferenceEndpointError as e:
                    instance_type, instance_size = InferenceEndpointModel.get_larger_hardware_suggestion(
                        instance_type, instance_size
                    )
                    must_scaleup_endpoint = True

                    logger.info(
                        f"Endpoint failed to start on current hardware with error {e}. Trying to autoscale to ({instance_type}, {instance_size})."
                    )
                except InferenceEndpointTimeoutError as e:
                    logger.error(
                        "Endpoint did not start within 30 minutes, there was a timeout. Please inspect the logs."
                    )
                    self.cleanup()
                    raise e
                except HfHubHTTPError as e:
                    # The endpoint actually already exists, we'll spin it up instead of trying to create a new one
                    if "409 Client Error: Conflict for url:" in str(e):
                        config.endpoint_name = endpoint_name
                        config.reuse_existing = True
                    # Requested resources are not available
                    elif "Bad Request: Compute instance not available yet" in str(e):
                        logger.error(
                            f"The hardware combination you are requesting does not seem to be available: ({instance_type}, {instance_size}, {config.region})."
                        )
                        raise e
                    # User account does not have access to requested resources
                    elif "Conflict: Quota exceeded" in str(e):
                        raise e
                except ConnectionError as e:
                    logger.error(f"Connection failed with error {e}. Retrying")

            if not self.endpoint.status == "running":
                raise Exception("Did not manage to start endpoint within the elapsed time and on suggested hardware.")

            logger.info("Endpoint successfully deployed!")
            self.endpoint_name = config.endpoint_name
            self.name = self.endpoint.repository
            self.revision = self.endpoint.revision
            self.async_client: AsyncInferenceClient = self.endpoint.async_client
            self.client: InferenceClient = self.endpoint.client

        else:  # Free inference client
            self.endpoint = None
            self.endpoint_name = None
            self.name = config.model_name
            self.revision = "default"
            self.async_client = AsyncInferenceClient(model=config.model_name, token=env_config.token)
            self.client = InferenceClient(model=config.model_name, token=env_config.token)

        self.use_async = True  # set to False for debug - async use is faster

        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False

        self.model_info = ModelInfo(
            model_name=self.name,
            model_sha=self.revision,
            model_dtype=getattr(config, "model_dtype", "default"),
            model_size=-1,
        )
        self.generation_parameters = config.generation_parameters
        self.generation_config = TextGenerationInputGenerateParameters(**self.generation_parameters.to_tgi_ie_dict())

    @staticmethod
    def get_larger_hardware_suggestion(cur_instance_type: str = None, cur_instance_size: str = None):
        cur_instance_ix = -1
        try:
            if cur_instance_type and cur_instance_size:
                cur_instance_ix = SORTED_INSTANCE_SIZES.index((cur_instance_type, cur_instance_size))
            new_instance_type = SORTED_INSTANCE_SIZES[cur_instance_ix + 1][0]
            new_instance_size = SORTED_INSTANCE_SIZES[cur_instance_ix + 1][1]
            return new_instance_type, new_instance_size
        except ValueError:
            raise Exception(
                f"Problem when scaling endpoint: the current instance combination ({cur_instance_type}, {cur_instance_size}) is unknown. Can't scale it up."
            )
        except IndexError:
            raise Exception(
                "To avoid accidental costs, we do not upgrade the current endpoint above 4 a100 automatically, please request it explicitely."
            )

    @staticmethod
    def get_suggested_model_config(model_repo):
        # Code from https://huggingface.co/spaces/huggingface/dedicated-endpoint-snooper/blob/main/app.py
        # Example of the suggestedCompute value: 'aws-us-east-1-nvidia-l4-x1'
        # -> aws us-east-1 nvidia-l4 x1
        url = f"https://ui.endpoints.huggingface.co/api/configuration?model_id={model_repo}"
        response = requests.get(url)
        config = response.json()

        suggested_compute = config["suggestedCompute"]
        suggested_vendor = suggested_compute.split("-")[0]
        if suggested_vendor == "azure":
            suggested_region = suggested_compute.split("-")[1]
        else:
            suggested_region = "-".join(suggested_compute.split("-")[1:4])
        suggested_instance = "-".join(suggested_compute.split("-")[-3:-1])
        suggested_size = suggested_compute.split("-")[-1]
        return suggested_vendor, suggested_region, suggested_instance, suggested_size

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def disable_tqdm(self) -> bool:
        False  # no accelerator = this is the main process

    def cleanup(self):
        if self.endpoint is not None:
            if self.reuse_existing:
                self.endpoint.pause()
                logger.warning(
                    "Since your endpoint was existing before, we did not delete it, but paused it instead. You might want to delete it if you're done using it."
                )
            else:
                self.endpoint.delete()
                logger.warning(
                    "We deleted the spinned up endpoint after using it. You'll need to create it again if you need to reuse it."
                )

    @property
    def max_length(self):
        if self._max_length is not None:
            return self._max_length

        if hasattr(self.tokenizer, "model_max_length"):
            self._max_length = self.tokenizer.model_max_length
        else:
            self._max_length = 2048
        return self._max_length

    def _async_process_request(
        self,
        context: str,
        stop_tokens: list[str],
        max_tokens: int,
        grammar: Optional[TextGenerationInputGrammarType] = None,
    ) -> Coroutine[None, list[TextGenerationOutput], str]:
        # Todo: add an option to launch with conversational instead for chat prompts
        # https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient.conversational
        generation_config: TextGenerationInputGenerateParameters = replace(
            self.generation_config,
            stop=stop_tokens,
            max_new_tokens=max_tokens,
            details=True,
            decoder_input_details=True,
            grammar=grammar,
        )

        generated_text = self.async_client.text_generation(prompt=context, generation_config=generation_config)

        return generated_text

    def _process_request(
        self,
        context: str,
        stop_tokens: list[str],
        max_tokens: int,
        grammar: Optional[TextGenerationInputGrammarType] = None,
    ) -> TextGenerationOutput:
        # Todo: add an option to launch with conversational instead for chat prompts
        # https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient.conversational
        generation_config: TextGenerationInputGenerateParameters = replace(
            self.generation_config,
            stop=stop_tokens,
            max_new_tokens=max_tokens,
            details=True,
            decoder_input_details=True,
            grammar=grammar,
        )

        generated_text = self.client.text_generation(
            prompt=context,
            generation_config=generation_config,
        )

        return generated_text

    async def _async_process_batch_generate(
        self,
        requests: list[GreedyUntilRequest],
    ) -> list[TextGenerationOutput]:
        return await asyncio.gather(
            *[
                self._async_process_request(
                    context=request.context,
                    stop_tokens=as_list(request.stop_sequence),
                    max_tokens=request.generation_size,
                    grammar=request.generation_grammar,
                )
                for request in requests
            ]
        )

    def _process_batch_generate(
        self,
        requests: list[GreedyUntilRequest],
    ) -> list[TextGenerationOutput]:
        return [
            self._process_request(
                context=request.context,
                stop_tokens=as_list(request.stop_sequence),
                max_tokens=request.generation_size,
                grammar=request.generation_grammar,
            )
            for request in requests
        ]

    async def _async_process_batch_logprob(
        self, requests: list[LoglikelihoodRequest], rolling: bool = False
    ) -> list[TextGenerationOutput]:
        return await asyncio.gather(
            *[
                self._async_process_request(
                    context=request.context if rolling else request.context + request.choice,
                    stop_tokens=[],
                    max_tokens=1,
                )
                for request in requests
            ]
        )

    def _process_batch_logprob(
        self, requests: list[LoglikelihoodRequest], rolling: bool = False
    ) -> list[TextGenerationOutput]:
        return [
            self._process_request(
                context=request.context if rolling else request.context + request.choice,
                stop_tokens=[],
                max_tokens=1,
            )
            for request in requests
        ]

    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> List[GenerativeResponse]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                # the `returns_logits` flag is only used to filter the results, we always request the full details.
                returns_logits = batch[0].use_logits
                num_samples = batch[0].num_samples
                if num_samples > 1:
                    logger.error(
                        "Inference endpoints does not allow sampling evaluations - this is likely to fail or provide problematic results"
                    )

                if self.use_async:
                    responses = asyncio.run(self._async_process_batch_generate(batch))
                else:
                    responses = self._process_batch_generate(batch)
                for i, response in enumerate(responses):
                    results.append(
                        GenerativeResponse(
                            result=response.generated_text,
                            logits=[item.logprob for item in response.details.prefill] if returns_logits else None,
                            generated_tokens=[token.id for token in response.details.tokens],
                            truncated_tokens_count=max(
                                len(self.tokenizer.encode(batch[i].context)) - self.max_length, 0
                            ),
                            padded_tokens_count=-1,
                        )
                    )

        return dataset.get_original_order(results)

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.tokenized_continuation = self.tok_encode(request.choice)
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(dataloader, desc="Loglikelihoods", position=1, leave=False, disable=self.disable_tqdm):
                if self.use_async:
                    responses = asyncio.run(self._async_process_batch_logprob(batch))
                else:
                    responses = self._process_batch_logprob(batch)
                for cur_request, response in zip(batch, responses):
                    cont_toks = torch.tensor(cur_request.tokenized_continuation)
                    len_choice = len(cont_toks)

                    if self.endpoint:  # inference endpoint
                        logits = [
                            t.logprob for t in response.details.prefill[-len_choice:] if t.logprob is not None
                        ]  # to check
                    else:  # serverless endpoint
                        logits = [t.logprob for t in response.details.tokens[-len_choice:] if t.logprob is not None]

                    greedy_tokens = torch.tensor(logits).argmax(dim=-1)
                    max_equal = (greedy_tokens == cont_toks).all().squeeze(0)
                    results.append(
                        LoglikelihoodResponse(
                            result=(sum(logits), bool(max_equal)),
                            input_tokens=[t.id for t in response.details.prefill[:-len_choice]],
                            generated_tokens=[t.id for t in response.details.prefill[-len_choice:]],
                            truncated_tokens_count=-1,
                            padded_tokens_count=-1,
                        )
                    )

        return dataset.get_original_order(results)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs=None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        for request in requests:
            request.tokenized_context = [self.tokenizer.eos_token_id]
            request.tokenized_continuation = self.tok_encode(request.context)

        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Loglikelihoods, rolling", position=1, leave=False, disable=self.disable_tqdm
            ):
                if self.use_async:
                    responses = asyncio.run(self._async_process_batch_logprob(batch, rolling=True))
                else:
                    responses = self._process_batch_logprob(batch, rolling=True)
                for response in responses:
                    logits = [t.logprob for t in response.details.tokens[:-1]]

                    results.append(
                        LoglikelihoodResponse(
                            result=sum(logits),
                            input_tokens=[t.id for t in response.details.prefill],
                            generated_tokens=[t.id for t in response.details.tokens[:-1]],
                            truncated_tokens_count=-1,
                            padded_tokens_count=-1,
                        )
                    )

        return dataset.get_original_order(results)

    def loglikelihood_single_token(
        self,
        requests: list[LoglikelihoodSingleTokenRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodSingleTokenResponse]:
        raise ValueError("Endpoint models can't use single token metrics. Change the metric to the standard version")

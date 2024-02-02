import asyncio
from typing import Coroutine, List, Optional, Union

from huggingface_hub import AsyncInferenceClient, InferenceEndpoint, create_inference_endpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.models.model_config import EnvConfig, InferenceEndpointModelConfig, InferenceModelConfig
from lighteval.models.model_output import GenerateReturn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    GreedyUntilWithLogitsRequest,
)


DATASET_SPLITS = 4
BATCH_SIZE = 50


class InferenceEndpointModel:
    def __init__(
        self, config: Union[InferenceEndpointModelConfig, InferenceModelConfig], env_config: EnvConfig
    ) -> None:
        if isinstance(config, InferenceEndpointModelConfig):
            self.endpoint: InferenceEndpoint = create_inference_endpoint(
                name=config.name,
                repository=config.repository,
                framework=config.framework,
                task="text-generation",
                accelerator=config.accelerator,
                vendor=config.vendor,
                region=config.region,
                type=config.endpoint_type,
                instance_size=config.instance_size,
                instance_type=config.instance_type,
                token=env_config.token,
                custom_image={
                    "health_route": "/health",
                    "env": {
                        "MAX_BATCH_PREFILL_TOKENS": "2048",
                        "MAX_INPUT_LENGTH": "1024",
                        "MAX_TOTAL_TOKENS": "1512",
                        "MODEL_ID": "/repository",
                    },
                    "url": "ghcr.io/huggingface/text-generation-inference:1.1.0",
                },
            )
            self.endpoint.wait()  # Waits for the endpoint to be deployed
            self.client: AsyncInferenceClient = self.endpoint.async_client

        else:  # Free inference client
            self.endpoint = None
            self.client = AsyncInferenceClient(model=config.model, token=env_config.token)

    def pause(self):
        if self.endpoint is not None:
            self.endpoint.scale_to_zero()

    def delete(self):
        if self.endpoint is not None:
            self.endpoint.delete()
            hlog_warn(
                "You deleted your endpoint after using it. You'll need to create it again if you need to reuse it."
            )

    def __process_request_generate(
        self, context: str, stop_tokens: list[str], max_tokens: int, returns_logits: bool
    ) -> Coroutine[None, List, str]:
        generated_text = self.client.text_generation(
            prompt=context,
            details=returns_logits,
            max_new_tokens=max_tokens,
            # truncate=,
            decoder_input_details=True,
            stop_sequences=stop_tokens,
        )

        return generated_text

    async def __process_batch_generate(self, batch, returns_logits: bool = False):
        stop_tokens, max_generated_tokens = batch[0][1]
        contexts = [c[0] for c in batch]

        return await asyncio.gather(
            *[
                self.__process_request_generate(
                    context=context,
                    stop_tokens=stop_tokens,
                    max_tokens=max_generated_tokens,
                    returns_logits=returns_logits,
                )
                for context in contexts
            ]
        )

    def greedy_until_with_logits(
        self,
        requests: list[GreedyUntilWithLogitsRequest],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
        dataset_splits: int = 4,
    ) -> list[GenerateReturn]:
        """
        Generates sequences greedily until a stopping condition is met,
        returning both the generated sequences and the logits.

        Args:
            requests (list[tuple[str, dict]]): A list of input requests,
                where each request is a tuple containing a prompt string and a dictionary of additional parameters.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
            override_bs (Optional[int], optional): Overrides the batch size for generation. Defaults to None.
            dataset_splits (int, optional): Number of splits to divide the dataset into for parallel generation. Defaults to 4.

        Returns:
            list[GenerateReturn]: A list of GenerateReturn objects,
                where each object contains the generated sequence and the corresponding logits.
        """

        return self.greedy_until(
            requests,
            returns_logits=True,
            disable_tqdm=disable_tqdm,
            override_bs=override_bs,
            dataset_splits=dataset_splits,
        )

    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        returns_logits: bool = False,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[str]:
        inputs = [
            (
                request.context,
                (request.stop_sequence + [self.eot_token], request.generation_size),
            )
            for request in requests
        ]
        dataset = GenerativeTaskDataset(requests=inputs, dataset_splits=DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(), total=DATASET_SPLITS, desc="Splits", position=0, disable=disable_tqdm
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(dataloader, desc="Greedy generation", position=1, leave=False, disable=disable_tqdm):
                responses = asyncio.run(self.__process_batch_generate(batch, returns_logits))
                for response in responses:
                    results.append(
                        GenerateReturn(
                            result=response.generated_text,
                        )
                    )

        return results

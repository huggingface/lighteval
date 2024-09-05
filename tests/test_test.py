import time
import random
import asyncio
from typing import Iterator

import pytest
import docker
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from huggingface_hub import (
    InferenceClient,
    AsyncInferenceClient,
    TextGenerationOutput,
)


@pytest.fixture(params=["sync", "async"])
def tgi_client(request) -> Iterator[InferenceClient|AsyncInferenceClient]:
    client = docker.from_env()

    try:
        container = client.containers.get("lighteval-tgi-model-test")
        port = container.ports["80/tcp"][0]["HostPort"]
    except docker.errors.NotFound:
        port = random.randint(8000, 9000)
        container = client.containers.run(
            "ghcr.io/huggingface/text-generation-inference:2.2.0",
            command=[
                "--model-id",
                "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "--dtype",
                "float16",
            ],
            detach=True,
            name="lighteval-tgi-model-test",
            auto_remove=False,
            ports={"80/tcp": port},
        )
    address = f"http://localhost:{port}"
    for _ in range(40):
        try:
            if requests.get(f"{address}/health"):
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Couldn't setup TGI server.")
    
    if request.param == "async":
        yield AsyncInferenceClient(base_url=address)    
    elif request.param == "sync":
        yield InferenceClient(base_url=address)
    else:
        raise RuntimeError()


def test_logprobs(tgi_client: InferenceClient|AsyncInferenceClient):
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    
    # It raises error in async setting unless the size of `prompts` is < 3
    prompts = [
        "Tell me:\n\nHow are you?Fine, thanks!",
        "Tell me:\n\nHow are you?Not bad!",
        "Tell me:\n\nComment vas-tu?Comme ci, comme ça",
        "Tell me:\n\nComment vas-tu?Ca va! Merci!",
    ]
    responses = []
    for prompt in prompts:
        responses.append(tgi_client.text_generation(
            prompt,
            details=True,
            decoder_input_details=True,
            max_new_tokens=1,
            stop_sequences=None,
            do_sample=False,
            return_full_text=False,
            seed=42,
        ))
    if isinstance(tgi_client, AsyncInferenceClient):
        loop = asyncio.get_event_loop()
        responses: list[TextGenerationOutput] = loop.run_until_complete(asyncio.gather(*responses))
    
    error = False
    for prompt, response in zip(prompts, responses):

        tgi_logprobs = torch.tensor([t.logprob for t in response.details.prefill[1:]]) # Skipping <s> whose logprob is None
        
        tokenized_sequence = tokenizer(prompt, return_tensors='pt')['input_ids']
        output = model.generate(tokenized_sequence, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True)
        with torch.no_grad():
            logprobs = torch.log_softmax(model.lm_head(output.hidden_states[0][-1]),dim=-1)
        logprobs = logprobs.gather(dim=-1, index=tokenized_sequence[:,1:].unsqueeze(-1)).squeeze()
        
        if not torch.allclose(logprobs.sum(), tgi_logprobs.sum()):
            print(f"====== prompt: {repr(prompt)} ======")
            print("TGI logprobs:", tgi_logprobs.tolist())
            print("TGI tokens:",[t.id for t in response.details.prefill])
            print("Ref. logprobs:", logprobs.tolist())
            print("Ref. tokens:", tokenized_sequence[0].tolist())
            error = True
    assert not error
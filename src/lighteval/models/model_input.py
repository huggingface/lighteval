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

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class GenerationParameters:
    early_stopping: Optional[bool] = None  # vllm, transformers
    repetition_penalty: Optional[float] = None  # vllm, transformers, tgi, sglang
    frequency_penalty: Optional[float] = None  # vllm, tgi, sglang
    length_penalty: Optional[float] = None  # vllm, transformers
    presence_penalty: Optional[float] = None  # vllm, sglang

    max_new_tokens: Optional[int] = None  # vllm, transformers, tgi, litellm, sglang
    min_new_tokens: Optional[int] = None  # vllm, transformers, sglang

    seed: Optional[int] = None  # vllm, tgi, litellm
    stop_tokens: Optional[list[str]] = None  # vllm, transformers, tgi, litellm, sglang
    temperature: Optional[float] = None  # vllm, transformers, tgi, litellm, sglang
    top_k: Optional[int] = None  # vllm, transformers, tgi, sglang
    min_p: Optional[float] = None  # vllm, transformers, sglang
    top_p: Optional[int] = None  # vllm, transformers, tgi, litellm, sglang
    truncate_prompt: Optional[bool] = None  # vllm, tgi

    # response format to be followed by the model,
    # more info here https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
    response_format: Optional[str] = None  # inference_providers

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Creates a GenerationParameters object from a config dictionary

        Args:
            config_dict (dict): Config dictionary. Must obey the following shape:
            {"generation":
                {
                    "early_stopping": value,
                    ...
                    "truncate_prompt": value
                }
            }
        """
        return GenerationParameters(**config_dict.get("generation", {}))

    @classmethod
    def from_model_args(cls, model_args: str):
        """Creates a GenerationParameters object from a model_args string.

        It's used when the model_args are passed as a string in the command line.
        The generation parameters must follow the following format (at any place in the string):
        "generation_parameters={key1:value1,key2=value2}"

        Args:
            model_args (str): A string like the following:
                "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,max_model_length=32768,generation={temperature:0.7,top_p:5}"
        """

        def parse_model_args(model_args):
            import json
            import re

            pattern = re.compile(r"(\w+)=(\{.*\}|[^,]+)")
            matches = pattern.findall(model_args)
            for key, value in matches:
                key = key.strip()
                if key == "generation_parameters":
                    gen_params = re.sub(r"(\w+):", r'"\1":', value)
                    return json.loads(gen_params)

        params: dict = parse_model_args(model_args) or {}
        return GenerationParameters(**params)

    def to_litellm_dict(self) -> dict:
        """Selects relevant generation and sampling parameters for litellm models.
        Doc: https://docs.litellm.ai/docs/completion/input#input-params-1

        Returns:
            dict: The parameters to create a litellm.SamplingParams in the model config.
        """
        args = {
            "max_completion_tokens": self.max_new_tokens,
            "stop": self.stop_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        return {k: v for k, v in args.items() if v is not None}

    def to_inference_providers_dict(self) -> dict:
        """Selects relevant generation and sampling parameters for litellm models.
        Doc: https://docs.litellm.ai/docs/completion/input#input-params-1

        Returns:
            dict: The parameters to create a litellm.SamplingParams in the model config.
        """
        args = {
            "top_p": self.top_p,
            "temperature": self.temperature,
            "stop": self.stop_tokens,
            "seed": self.seed,
            "response_format": self.response_format,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_new_tokens,
            "logprobs": None,
            "logit_bias": None,
            "frequency_penalty": self.frequency_penalty,
        }
        return {k: v for k, v in args.items() if v is not None}

    def to_vllm_dict(self) -> dict:
        """Selects relevant generation and sampling parameters for vllm models.
        Doc: https://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html

        Returns:
            dict: The parameters to create a vllm.SamplingParams in the model config.
        """
        sampling_params_to_vllm_naming = {
            "max_new_tokens": "max_tokens",
            "min_new_tokens": "min_tokens",
            "stop_tokens": "stop",
        }

        # Task specific sampling params to set in model: n, best_of, use_beam_search
        # Generation specific params to set in model: logprobs, prompt_logprobs
        x = {sampling_params_to_vllm_naming.get(k, k): v for k, v in asdict(self).items() if v is not None}
        # VLLM max_tokens is 16 by default, however the pipeline expect the max_tokens to be None, if the user didn't specify it
        if not x.get("max_tokens"):
            x["max_tokens"] = None
        return x

    def to_vllm_openai_dict(self) -> dict:
        """Selects relevant generation and sampling parameters for vllm and openai models.
        Doc: https://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html

        Returns:
            dict: The parameters to create a vllm.SamplingParams or just provide OpenAI params as such in the model config.
        """
        # Task specific sampling params to set in model: n, best_of, use_beam_search
        # Generation specific params to set in model: logprobs, prompt_logprobs
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_transformers_dict(self) -> dict:
        """Selects relevant generation and sampling parameters for transformers models.
        Doc: https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/text_generation#transformers.GenerationConfig

        Note: We actually don't use the GenerationConfig object itself because it has a huge number of parameters automatically
        initialized, to a config which slows down evals insanely.

        Returns:
            dict: The parameters to create a transformers.GenerationConfig in the model config.
        """
        # Task specific sampling params to set in model: do_sample, num_return_sequences, num_beans
        args = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "early_stopping": self.early_stopping,
            "stop_strings": self.stop_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        return {k: v for k, v in args.items() if v is not None}

    def to_tgi_ie_dict(self) -> dict:
        """Selects relevant generation and sampling parameters for tgi or inference endpoints models.
        Doc: https://huggingface.co/docs/huggingface_hub/v0.26.3/en/package_reference/inference_types#huggingface_hub.TextGenerationInputGenerateParameters

        Returns:
            dict: The parameters to create a huggingface_hub.TextGenerationInputGenerateParameters in the model config.
        """
        # Task specific sampling params to set in model: best_of, do_sample
        args = {
            "decoder_input_details": True,
            "details": True,
            "frequency_penalty": self.frequency_penalty,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "stop": self.stop_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "truncate": self.truncate_prompt,
        }
        return {k: v for k, v in args.items() if v is not None}

    def to_sglang_dict(self) -> dict:
        args = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stop": self.stop_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "min_new_tokens": self.min_new_tokens,
        }
        return {k: v for k, v in args.items() if v is not None}

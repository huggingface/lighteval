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
    repetition_penalty: Optional[float] = None  # vllm, transformers, tgi
    frequency_penalty: Optional[float] = None  # vllm, tgi
    length_penalty: Optional[float] = None  # vllm, transformers
    presence_penalty: Optional[float] = None  # vllm

    max_new_tokens: Optional[int] = None  # vllm, transformers, tgi
    min_new_tokens: Optional[int] = None  # vllm, transformers

    seed: Optional[int] = None  # vllm, tgi
    stop_tokens: Optional[list[str]] = None  # vllm, transformers, tgi
    temperature: Optional[float] = None  # vllm, transformers, tgi
    top_k: Optional[int] = None  # vllm, transformers, tgi
    min_p: Optional[float] = None  # vllm, transformers
    top_p: Optional[int] = None  # vllm, transformers, tgi
    truncate_prompt: Optional[bool] = None  # vllm, tgi

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

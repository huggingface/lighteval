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

from dataclasses import dataclass
from typing import Optional

from huggingface_hub import TextGenerationInputGrammarType

from lighteval.utils.imports import NO_VLLM_ERROR_MSG, is_vllm_available


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

    grammar: Optional[TextGenerationInputGrammarType] = None  # tgi

    def to_vllm(self):
        if not is_vllm_available():
            raise ImportError(NO_VLLM_ERROR_MSG)
        from vllm import SamplingParameters

        # Task specific sampling params to set in model: n, best_of, use_beam_search
        # Generation specific params to set in model: logprobs, prompt_logprobs
        args = {
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "seed": self.seed,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "stop": self.stop_tokens,
            "max_tokens": self.max_new_tokens,
            "min_tokens": self.min_new_tokens,
            "truncate_prompt_tokens": self.truncate_prompt,
        }
        return SamplingParameters(**{k: v for k, v in args.items() if v is not None})

    def to_transformers(self):
        from transformers import GenerationConfig

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
        # Even though we only use the dict representation of the GenerationConfig
        # we still create the object as it uses validation steps
        return GenerationConfig(**{k: v for k, v in args.items() if v is not None})

    def to_tgi(self):
        from huggingface_hub import TextGenerationInputGenerateParameters

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
            "grammar": self.grammar,
        }
        return TextGenerationInputGenerateParameters(**{k: v for k, v in args.items() if v is not None})

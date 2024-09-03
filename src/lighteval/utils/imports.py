# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib


def is_accelerate_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


NO_ACCELERATE_ERROR_MSG = "You requested the use of accelerate for this evaluation, but it is not available in your current environement. Please install it using pip."


def is_tgi_available() -> bool:
    return importlib.util.find_spec("text_generation") is not None


NO_TGI_ERROR_MSG = "You are trying to start a text generation inference endpoint, but text-generation is not present in your local environement. Please install it using pip."


def is_nanotron_available() -> bool:
    return importlib.util.find_spec("nanotron") is not None


NO_NANOTRON_ERROR_MSG = "You requested the use of nanotron for this evaluation, but it is not available in your current environement. Please install it using pip."


def is_optimum_available() -> bool:
    return importlib.util.find_spec("optimum") is not None


def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


NO_BNB_ERROR_MSG = "You are trying to load a model quantized with `bitsandbytes`, which is not available in your local environement. Please install it using pip."


def is_autogptq_available() -> bool:
    return importlib.util.find_spec("auto_gptq") is not None


NO_AUTOGPTQ_ERROR_MSG = "You are trying to load a model quantized with `auto-gptq`, which is not available in your local environement. Please install it using pip."


def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None


NO_PEFT_ERROR_MSG = "You are trying to use adapter weights models, for which you need `peft`, which is not available in your environment. Please install it using pip."


def is_tensorboardX_available() -> bool:
    return importlib.util.find_spec("tensorboardX") is not None


NO_TENSORBOARDX_WARN_MSG = (
    "You are trying to log using tensorboardX, which is not installed. Please install it using pip. Skipping."
)


def is_openai_available() -> bool:
    return importlib.util.find_spec("openai") is not None


NO_OPENAI_ERROR_MSG = "You are trying to use an Open AI LLM as a judge, for which you need `openai`, which is not available in your environment. Please install it using pip."


def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None and importlib.util.find_spec("ray") is not None


NO_VLLM_ERROR_MSG = "You are trying to use an VLLM model, for which you need `vllm` and `ray`, which are not available in your environment. Please install them using pip, `pip install vllm ray`."


def can_load_extended_tasks() -> bool:
    imports = []
    for package in ["langdetect", "openai"]:
        imports.append(importlib.util.find_spec(package))

    return all(cur_import is not None for cur_import in imports)


CANNOT_USE_EXTENDED_TASKS_MSG = "If you want to use extended_tasks, make sure you installed their dependencies using `pip install -e .[extended_tasks]`."
